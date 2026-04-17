// rocm-cpp C API implementation — wraps CK's DeviceGemm_Wmma_CShuffleV3 and
// exposes a C interface for consumers (halo-1bit, lemond, external).
//
// The CK surface is fully contained in this TU. Consumers need only the C
// header and librocm_cpp.so.

#include "rocm_cpp/ck_gemm.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace {

using ck::half_t;
using ck::pk_i4_t;
using ck::index_t;

template <index_t... Is> using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;

// Winning instance from tile-tuning sweep: BlockSize=256, 128x128x32 tile,
// Interwave v1 pipeline, PermuteB=true. 0.96x rocBLAS FP16 at half B memory
// on BitNet FFN shapes (see docs/10-ck-integration-path.md).
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
    Row, Col, Row,
    half_t, pk_i4_t, half_t, float, half_t,
    PassThrough, PassThrough, PassThrough, GemmDefault,
    256,
    128, 128, 32,
    8, 8,
    16, 16,
    4, 2,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    1, 1, S<1, 32, 1, 8>, 8,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1,
    half_t, half_t, /*PermuteA=*/false, /*PermuteB=*/true>;

constexpr int KPerBlock = 32;

}  // namespace

struct rcpp_ck_gemm_handle {
    int M, N, K;
    DeviceGemmInstance gemm{};
    DeviceGemmInstance::Invoker invoker = gemm.MakeInvoker();
    std::string instance_str;
};

extern "C" {

rcpp_status_t
rcpp_ck_gemm_create(int M, int N, int K, rcpp_ck_gemm_handle_t** handle_out) {
    if(!handle_out || M <= 0 || N <= 0 || K <= 0) return RCPP_INVALID_ARG;
    if(K % KPerBlock != 0)                        return RCPP_INVALID_ARG;

    auto* h = new(std::nothrow) rcpp_ck_gemm_handle{M, N, K};
    if(!h) return RCPP_INTERNAL;

    // Validate that CK accepts this shape (probe with a dummy argument).
    auto dummy_arg = h->gemm.MakeArgument(
        /*A*/ static_cast<const half_t*>(nullptr),
        /*B*/ static_cast<const pk_i4_t*>(nullptr),
        /*C*/ static_cast<half_t*>(nullptr),
        M, N, K,
        /*StrideA*/ K,
        /*StrideB*/ K,
        /*StrideC*/ N,
        /*KBatch*/ 1,
        PassThrough{}, PassThrough{}, PassThrough{});
    if(!h->gemm.IsSupportedArgument(dummy_arg)) {
        delete h;
        return RCPP_UNSUPPORTED;
    }

    h->instance_str = h->gemm.GetTypeString();
    *handle_out = h;
    return RCPP_OK;
}

void
rcpp_ck_gemm_destroy(rcpp_ck_gemm_handle_t* h) {
    delete h;
}

rcpp_status_t
rcpp_ck_gemm_run(rcpp_ck_gemm_handle_t* h,
                 const void* A_dev, const void* B_dev_packed, void* C_dev,
                 void* stream) {
    if(!h || !A_dev || !B_dev_packed || !C_dev) return RCPP_INVALID_ARG;

    auto arg = h->gemm.MakeArgument(
        static_cast<const half_t*>(A_dev),
        static_cast<const pk_i4_t*>(B_dev_packed),
        static_cast<half_t*>(C_dev),
        h->M, h->N, h->K,
        h->K, h->K, h->N,
        /*KBatch*/ 1,
        PassThrough{}, PassThrough{}, PassThrough{});

    if(!h->gemm.IsSupportedArgument(arg)) return RCPP_UNSUPPORTED;

    StreamConfig cfg{static_cast<hipStream_t>(stream), /*time_kernel=*/false, 0};
    h->invoker.Run(arg, cfg);
    return RCPP_OK;
}

const char*
rcpp_ck_gemm_instance_string(const rcpp_ck_gemm_handle_t* h) {
    return h ? h->instance_str.c_str() : "";
}

// Forward-declared launchers from src/prefill_standalone.hip
extern void rcpp_standalone_launch_wmma_4x4_vec(const void*, const void*, void*,
                                                int, int, int, void*);

rcpp_status_t
rcpp_standalone_gemm(const void* A_dev, const void* B_dev_packed, void* C_dev,
                     int M, int N, int K, void* stream) {
    if(!A_dev || !B_dev_packed || !C_dev) return RCPP_INVALID_ARG;
    if(M <= 0 || N <= 0 || K <= 0)        return RCPP_INVALID_ARG;
    if(M % 64 != 0 || N % 64 != 0)        return RCPP_INVALID_ARG;
    if(K % 32 != 0)                        return RCPP_INVALID_ARG;
    // Phase 4i kernel: 64x64 output + vectorized A loads. 4% over Phase 4h on
    // 4096^3 square, parity on BitNet FFN. Best standalone across all shapes.
    rcpp_standalone_launch_wmma_4x4_vec(A_dev, B_dev_packed, C_dev, M, N, K, stream);
    return RCPP_OK;
}

// Offline packer — ternary int8 {-1, 0, +1} col-major [K, N] -> pk_i4
// WMMA-permuted bytes [K*N/2].
//
// Byte-level semantics (replacement for CK's ck::Tensor<pk_i4_t> approach):
//
//   Un-permuted byte layout, col-major B:
//     For each column n in [0,N), each K-pair (k, k+1):
//       byte[(n*K + k)/2] = (hi << 4) | lo
//     where hi = nibble(ternary[n, k    ])  -- first element, HIGH nibble
//           lo = nibble(ternary[n, k + 1])  -- second element, LOW nibble
//     (High-first ordering is required by CK_USE_PK4_LAYOUT_SHUFFLE=1 set
//      in ck.hpp — see ck/utility/type_convert.hpp:type_convert<half2_t,pk_i4_t>.)
//
//   Nibble map compensates for CK's "n - 8" FP16 decode:
//     -1 -> 0x7  (decodes 7 - 8 = -1)
//      0 -> 0x8  (decodes 8 - 8 =  0)
//     +1 -> 0x9  (decodes 9 - 8 = +1)
//
//   Block permute (K -> [K0, N, K1] with K1 = KPerBlock = 32):
//     CK's reference uses 1-arg access on a pk_i4_t 2D tensor with NDEBUG;
//     resolves to a byte-for-byte memcpy at K1/2 byte granularity:
//       dst_byte = (j * N * K1/2) + (i * K1/2) + (jj/2)
//       src_byte = (i * K/2)      + (j * K1/2) + (jj/2)
//     With K1 = 32, that's a 16-byte contiguous copy per (j, i).
//
//   Within-8 nibble permute (01234567 -> 20643175, per upstream example):
//     For each column i in [0,N), each K-group of 8 (j step 8):
//       read the 4 bytes at byte-offset (i*K + j)/2
//       decompose 8 nibbles (high-first) into input[0..7]
//       write back 4 bytes with nibble permutation
//         byte[k+0,k+1] := (input[2]<<4)|input[0]
//         byte[k+2,k+3] := (input[6]<<4)|input[4]
//         byte[k+4,k+5] := (input[3]<<4)|input[1]
//         byte[k+6,k+7] := (input[7]<<4)|input[5]
//
// No CK headers, no libutility.a linkage — pure C++ with one std::vector.
rcpp_status_t
rcpp_ternary_pack_pk_i4(const int8_t* ternary_host, int8_t* packed_host,
                        int K, int N) {
    if(!ternary_host || !packed_host || K <= 0 || N <= 0) return RCPP_INVALID_ARG;
    if(K % KPerBlock != 0 || K % 8 != 0)                  return RCPP_INVALID_ARG;

    auto t_to_i4 = [](int8_t t) -> uint8_t {
        if(t == 0)  return 0x8;
        if(t >  0)  return 0x9;
        return 0x7;
    };

    const std::size_t nbytes = (std::size_t)K * N / 2;

    // Stage A: pack ternary -> un-permuted pk_i4 bytes (col-major [K, N]).
    std::vector<uint8_t> unpermuted(nbytes);
    for(int n = 0; n < N; ++n) {
        for(int k = 0; k < K; k += 2) {
            uint8_t hi = t_to_i4(ternary_host[(std::size_t)n * K + k    ]);
            uint8_t lo = t_to_i4(ternary_host[(std::size_t)n * K + k + 1]);
            unpermuted[((std::size_t)n * K + k) / 2] = (uint8_t)((hi << 4) | lo);
        }
    }

    // Stage B: block-reshape K -> [K0, N, K1] at K1/2-byte granularity into
    // the output buffer. After this the output holds the "block-permuted"
    // bytes; the within-8 nibble permute rewrites them in place.
    constexpr int K1       = KPerBlock;   // 32
    constexpr int K1_bytes = K1 / 2;      // 16
    const     int K0       = K / K1;
    uint8_t* dst = reinterpret_cast<uint8_t*>(packed_host);
    for(int j = 0; j < K0; ++j) {
        for(int i = 0; i < N; ++i) {
            const std::size_t dst_off =
                (std::size_t)j * N * K1_bytes + (std::size_t)i * K1_bytes;
            const std::size_t src_off =
                (std::size_t)i * (K / 2) + (std::size_t)j * K1_bytes;
            std::memcpy(dst + dst_off, unpermuted.data() + src_off, K1_bytes);
        }
    }

    // Stage C: within-8 nibble permute in place on the output bytes, using
    // the (j, i) indexing that CK's reference uses post-block-permute:
    //   byte-offset = (i * K + j) / 2, 4 bytes per group.
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < K; j += 8) {
            uint8_t* p = dst + ((std::size_t)i * K + j) / 2;  // 4 contiguous bytes
            int input[8];
            for(int kk = 0; kk < 4; ++kk) {
                int b = p[kk];
                input[kk * 2 + 0] = (b >> 4) & 0xf;
                input[kk * 2 + 1] = (b >> 0) & 0xf;
            }
            p[0] = (uint8_t)((input[2] << 4) | input[0]);
            p[1] = (uint8_t)((input[6] << 4) | input[4]);
            p[2] = (uint8_t)((input[3] << 4) | input[1]);
            p[3] = (uint8_t)((input[7] << 4) | input[5]);
        }
    }
    return RCPP_OK;
}

}  // extern "C"
