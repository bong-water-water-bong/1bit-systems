// rocm-cpp C API — BitNet-style ternary GEMM on gfx1151
//
// Batched prefill: FP16 activations [M, K] × ternary weights [K, N] -> FP16 [M, N].
// Weights are pre-packed once at model load via rcpp_ternary_pack_pk_i4 and
// stored in WMMA-permuted pk_i4 layout (K * N / 2 bytes, half of FP16).
//
// Consumers do NOT pull in CK or any HIP templates — only this C header.
// Link: librocm_cpp.so (+ libhip64, HIP runtime).

#ifndef ROCM_CPP_CK_GEMM_H
#define ROCM_CPP_CK_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RCPP_OK           = 0,
    RCPP_INVALID_ARG  = 1,
    RCPP_UNSUPPORTED  = 2,
    RCPP_HIP_ERROR    = 3,
    RCPP_INTERNAL     = 4,
} rcpp_status_t;

typedef struct rcpp_ck_gemm_handle rcpp_ck_gemm_handle_t;

// Create a handle for FP16 x pk_i4 -> FP16 GEMM at shape (M, N, K).
// Picks the CK instance internally; handle is reusable across stream launches.
// Returns RCPP_UNSUPPORTED if no CK instance accepts this shape (pad if needed).
rcpp_status_t
rcpp_ck_gemm_create(int M, int N, int K, rcpp_ck_gemm_handle_t** handle_out);

void
rcpp_ck_gemm_destroy(rcpp_ck_gemm_handle_t* handle);

// C[M, N] = A[M, K] * B[K, N] on the given HIP stream.
//   A_dev         : FP16 row-major [M, K] on device, stride K
//   B_dev_packed  : pk_i4 WMMA-permuted bytes [K*N/2] on device (from rcpp_ternary_pack_pk_i4)
//   C_dev         : FP16 row-major [M, N] on device, stride N
//   stream        : hipStream_t (void* to avoid a HIP include in the header)
rcpp_status_t
rcpp_ck_gemm_run(rcpp_ck_gemm_handle_t* handle,
                 const void* A_dev, const void* B_dev_packed, void* C_dev,
                 void* stream);

// Offline weight packer (host side).
//   ternary_host  : int8 values in {-1, 0, +1}, col-major [K, N], size K*N bytes
//   packed_host   : output, pk_i4 WMMA-permuted, size K*N/2 bytes
// Requires K % 32 == 0 and K % 8 == 0 (BitNet FFN / attention shapes satisfy this).
rcpp_status_t
rcpp_ternary_pack_pk_i4(const int8_t* ternary_host,
                        int8_t* packed_host,
                        int K, int N);

// Informational — returns CK's instance type string (or a stub if not built).
// Lifetime: tied to the handle.
const char*
rcpp_ck_gemm_instance_string(const rcpp_ck_gemm_handle_t* handle);

// -----------------------------------------------------------------------------
// Standalone (CK-free) prefill launcher.
//
// Same inputs as rcpp_ck_gemm_run. Produces bit-identical output to the CK
// backend on BitNet-realistic shapes; reaches 94% of CK's tuned WMMA perf on
// gfx1151 with ZERO ck/ includes in this TU (see src/prefill_standalone.hip,
// docs/11-de-ck-plan.md).
//
// Use this when you want the library to ship without the CK template surface —
// e.g., for a binary distribution that should not depend on TheRock being
// pre-built on the consumer's machine.
//
// Stateless: no handle needed. M, N, K must satisfy M % 64 == 0, N % 64 == 0,
// K % 32 == 0 for the 64x64 output-tile kernel; callers with arbitrary shapes
// should pad or fall back to the CK backend.
rcpp_status_t
rcpp_standalone_gemm(const void* A_dev, const void* B_dev_packed, void* C_dev,
                     int M, int N, int K, void* stream);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_CK_GEMM_H
