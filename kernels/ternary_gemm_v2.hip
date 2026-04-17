// Kernel v2.1 GEMM — Ternary prefill via rocWMMA INT8 (gfx1151, Wave32)
//
// Upgrades the v2 skeleton from a scalar-dot4 placeholder to the real
// 16x16x16 INT8 WMMA tile op. rocWMMA wraps the lane-layout sensitive
// intrinsic so we can focus on packing and staging rather than VGPR
// register allocation.
//
// Flow per output tile:
//   1. LDS-stage A_tile (16 rows x 16 k) of int8 weights, unpacked from
//      ternary bitstream.
//   2. LDS-stage B_tile (16 k x 16 batches) of int8 activations,
//      quantized from float with a per-tile act_scale.
//   3. mma_sync: C_frag += A_frag * B_frag
//   4. After K is consumed, rescale accumulator by act_scale * w_scale
//      and store back as float.
//
// Grid:  (M/16, B/16)
// Block: (32, 1, 1)  — one wave per 16x16 output tile
//
// "There and back again." — Bilbo

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_SIZE 32

__device__ __forceinline__ int quant_i8(float v, float inv_scale) {
    int q = __float2int_rn(v * inv_scale);
    return max(-127, min(127, q));
}

// One-wave ternary GEMM kernel producing a 16x16 output tile of Y.
// Y is laid out [B, M] row-major; the block's output tile is
//   Y[tile_b : tile_b+16, tile_m : tile_m+16].
__global__ void ternary_gemm_v2_kernel(
    const uint8_t* __restrict__ packed_bits,  // [M, K/8]
    const float*   __restrict__ x,            // [B, K]
    const float*   __restrict__ scales,       // [M]
    float*         __restrict__ y,            // [B, M]
    int M, int K, int B)
{
    const int tile_m = blockIdx.x * WMMA_M;
    const int tile_b = blockIdx.y * WMMA_N;
    if (tile_m >= M || tile_b >= B) return;

    const int tid = threadIdx.x;  // 0..31

    // LDS staging buffers.
    //   a_lds is row-major [16 rows x 16 k]  (weight tile)
    //   b_lds is col-major [16 k x 16 batches] (activation tile)
    // rocWMMA expects matrix_a row-major, matrix_b col-major for a standard
    // C = A * B where both are the "natural" orientation.
    __shared__ int8_t a_lds[WMMA_M * WMMA_K];
    __shared__ int8_t b_lds[WMMA_K * WMMA_N];
    __shared__ float  act_scale_sh;

    const int bytes_per_row = K / 8;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;
    fill_fragment(c_frag, 0);

    // Accumulate K in WMMA_K chunks. Each chunk is quantized independently
    // using a per-chunk activation scale. We then rescale into a float
    // accumulator at the end of the chunk.
    float float_acc[8] = {0};  // per-lane output cells

    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // --- Stage A: unpack weights into LDS a_lds[16 rows x 16 k] ---
        // 16 rows x 16 cols = 256 int8 values; 32 threads, 8 each.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < WMMA_M * WMMA_K) {
                int r = idx / WMMA_K;
                int k = idx % WMMA_K;
                int row = tile_m + r;
                int k_global = k_tile + k;
                int val = 0;
                if (row < M && k_global < K) {
                    int bit = (packed_bits[(size_t)row * bytes_per_row + k_global / 8]
                               >> (k_global % 8)) & 1;
                    val = bit ? 1 : -1;
                }
                a_lds[r * WMMA_K + k] = (int8_t)val;
            }
        }

        // --- Stage B: compute tile-local max-abs and quantize activations ---
        // First pass: find max abs over this tile's B_tile slice.
        // Each lane scans 8 positions of the 256-cell tile.
        float local_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < WMMA_K * WMMA_N) {
                int k = idx % WMMA_K;
                int b = idx / WMMA_K;
                int k_global = k_tile + k;
                int batch = tile_b + b;
                if (k_global < K && batch < B) {
                    float v = fabsf(x[(size_t)batch * K + k_global]);
                    if (v > local_max) local_max = v;
                }
            }
        }
        // Wave reduce max
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float other = __shfl_xor(local_max, off);
            if (other > local_max) local_max = other;
        }
        if (tid == 0) {
            act_scale_sh = (local_max > 1e-8f) ? (local_max / 127.0f) : 1.0f;
        }
        __syncthreads();
        float act_scale = act_scale_sh;
        float inv_scale = 1.0f / act_scale;

        // Second pass: quantize into b_lds[16 k x 16 batches] col-major.
        // Col-major means b_lds[k + batch * WMMA_K] stores B[k][batch].
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < WMMA_K * WMMA_N) {
                int k = idx % WMMA_K;
                int b = idx / WMMA_K;
                int k_global = k_tile + k;
                int batch = tile_b + b;
                int q = 0;
                if (k_global < K && batch < B) {
                    q = quant_i8(x[(size_t)batch * K + k_global], inv_scale);
                }
                b_lds[k + b * WMMA_K] = (int8_t)q;
            }
        }
        __syncthreads();

        // --- WMMA: local_c = A * B (int32 accumulator) ---
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> local_c;
        fill_fragment(local_c, 0);
        load_matrix_sync(a_frag, a_lds, WMMA_K);       // row-major, lda = 16
        load_matrix_sync(b_frag, b_lds, WMMA_K);       // col-major, ldb = 16
        mma_sync(local_c, a_frag, b_frag, local_c);

        // --- Fold into float accumulator with this chunk's act_scale ---
        #pragma unroll
        for (int i = 0; i < local_c.num_elements; i++) {
            float_acc[i] += (float)local_c.x[i] * act_scale;
        }
        __syncthreads();
    }

    // --- Store float_acc to Y with per-row weight scale ---
    // rocWMMA accumulator layout for Wave32 16x16: each lane owns 8 cells.
    // We can't directly store floats through rocwmma (it's int32 accumulator),
    // so stage c_frag through LDS back to memory using a reinterpret-friendly
    // path: cast float_acc into an int32 fragment for store_matrix_sync, then
    // have the block cooperatively write out with row scales.
    //
    // Simpler path: pack float_acc back into c_frag (as int32 reinterpret of
    // float bits is UB, so we do it through LDS with a float view).
    __shared__ float c_lds[WMMA_M * WMMA_N];

    // Move float_acc → c_lds using rocWMMA's lane layout. We piggyback on
    // store_matrix_sync for the int32 layout and index-match.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_store;
    #pragma unroll
    for (int i = 0; i < c_store.num_elements; i++) {
        // Encode float as int32 bit-pattern so store layout matches.
        int32_t bits;
        __builtin_memcpy(&bits, &float_acc[i], sizeof(float));
        c_store.x[i] = bits;
    }
    store_matrix_sync(reinterpret_cast<int32_t*>(c_lds), c_store, WMMA_N, mem_row_major);
    __syncthreads();

    // Apply per-row weight scale and write to global Y.
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < WMMA_M * WMMA_N) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int row = tile_m + r;
            int batch = tile_b + c;
            if (row < M && batch < B) {
                y[(size_t)batch * M + row] = c_lds[r * WMMA_N + c] * scales[row];
            }
        }
    }
}

extern "C" {

void ternary_gemm_v2(
    const uint8_t* packed_bits, const float* x, const float* scales,
    float* y, int M, int K, int B, hipStream_t stream)
{
    dim3 grid((M + WMMA_M - 1) / WMMA_M, (B + WMMA_N - 1) / WMMA_N);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(ternary_gemm_v2_kernel, grid, block, 0, stream,
        packed_bits, x, scales, y, M, K, B);
}

} // extern "C"
