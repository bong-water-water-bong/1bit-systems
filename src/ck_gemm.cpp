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

#include "ck/library/utility/host_tensor.hpp"

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

// Offline packer — ternary int8 {-1, 0, +1} col-major [K, N] -> pk_i4
// WMMA-permuted bytes [K*N/2]. Uses CK's Tensor type internally to apply the
// same PermuteB pipeline the GPU kernel expects.
rcpp_status_t
rcpp_ternary_pack_pk_i4(const int8_t* ternary_host, int8_t* packed_host,
                        int K, int N) {
    if(!ternary_host || !packed_host || K <= 0 || N <= 0) return RCPP_INVALID_ARG;
    if(K % KPerBlock != 0 || K % 8 != 0)                  return RCPP_INVALID_ARG;

    // Build CK tensors in pk_i4 layout: descriptor says [K, N] col-major but
    // storage is K*N/2 bytes (packed_size_v<pk_i4_t> = 2).
    using BDataType = pk_i4_t;

    // Ternary -> int4 nibble map, compensating for CK's "n - 8" decode
    // (see ck/utility/type_convert.hpp::type_convert<half2_t, pk_i4_t>).
    //   -1 -> 0x7  (decodes 7 - 8 = -1)
    //    0 -> 0x8  (decodes 8 - 8 =  0)
    //   +1 -> 0x9  (decodes 9 - 8 = +1)
    auto t_to_i4 = [](int8_t t) -> uint8_t {
        if(t == 0)  return 0x8;
        if(t >  0)  return 0x9;
        return 0x7;
    };

    // Source bytes: pack ternary pairs (col-major) into byte layout CK expects
    // for the un-permuted tensor. With CK_USE_PK4_LAYOUT_SHUFFLE defined
    // (default, set in ck.hpp), the HIGH nibble of each byte holds the FIRST
    // element (K=2k), LOW nibble holds the SECOND element (K=2k+1).
    ck::HostTensorDescriptor desc_knn(
        {static_cast<std::size_t>(K), static_cast<std::size_t>(N)},
        {static_cast<std::size_t>(1), static_cast<std::size_t>(K)});  // col-major stride {1, K}
    ck::Tensor<BDataType> b_k_n(desc_knn);
    {
        int8_t* dst = reinterpret_cast<int8_t*>(b_k_n.mData.data());
        for(int n = 0; n < N; ++n) {
            for(int k = 0; k < K; k += 2) {
                uint8_t hi = t_to_i4(ternary_host[(std::size_t)n * K + k    ]);  // first elem -> high nibble
                uint8_t lo = t_to_i4(ternary_host[(std::size_t)n * K + k + 1]);  // second elem -> low nibble
                dst[((std::size_t)n * K + k) / 2] = (int8_t)((hi << 4) | lo);
            }
        }
    }

    // Allocate b_k_n_permute with same descriptor.
    ck::Tensor<BDataType> b_k_n_permute(desc_knn);

    // Block-reshape [K, N] -> [K0, N, K1]  (exact replication of upstream).
    constexpr int K1 = KPerBlock;
    const int     K0 = K / K1;
    for(int j = 0; j < K0; ++j) {
        for(int i = 0; i < N; ++i) {
            for(int jj = 0; jj < K1; ++jj) {
                b_k_n_permute(j * N * K1 + i * K1 + jj) =
                    b_k_n(i * K + (j * K1 + jj));
            }
        }
    }

    // Within-8 nibble permute (01234567 -> 20643175). Exact replication of
    // gemm_wmma_fp16_pk_i4_v3.cpp lines 168-215.
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < K; j += 8) {
            int input[8];
            for(int kk = 0; kk < 4; ++kk) {
                int i4x2         = b_k_n_permute(j + kk * 2, i).data;
                input[kk * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[kk * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            {
                int hi   = input[2];
                int lo   = input[0];
                int i4x2 = (hi << 4) | lo;
                b_k_n_permute(j + 0, i) = (int8_t)i4x2;
            }
            {
                int hi   = input[6];
                int lo   = input[4];
                int i4x2 = (hi << 4) | lo;
                b_k_n_permute(j + 2, i) = (int8_t)i4x2;
            }
            {
                int hi   = input[3];
                int lo   = input[1];
                int i4x2 = (hi << 4) | lo;
                b_k_n_permute(j + 4, i) = (int8_t)i4x2;
            }
            {
                int hi   = input[7];
                int lo   = input[5];
                int i4x2 = (hi << 4) | lo;
                b_k_n_permute(j + 6, i) = (int8_t)i4x2;
            }
        }
    }

    // Copy packed bytes out. Tensor<pk_i4_t>.mData is a vector<pk_i4_t> which
    // is actually sizeof(int8_t) per element — storage is K*N/2 bytes but the
    // vector reports K*N elements. Each element is one BYTE (pk_i4_t::data).
    const int8_t* src = reinterpret_cast<const int8_t*>(b_k_n_permute.mData.data());
    std::memcpy(packed_host, src, (std::size_t)K * N / 2);
    return RCPP_OK;
}

}  // extern "C"
