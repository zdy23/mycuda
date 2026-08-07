#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>

// Online softmax fused attention:
//   O = softmax(Q @ K^T / sqrt(d)) @ V
// Q [M, d], K [N, d], V [N, d], O [M, d] — row-major.
// Single kernel, zero N×N buffer.  Grid-stride over queries, UNROLL for MLP.

#define UNROLL 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int threadsPerBlock>
__global__ void __launch_bounds__(threadsPerBlock)
softmax_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int M, int N, int d)
{
    int tid   = threadIdx.x;
    int num_threads = blockDim.x;
    int num_warps   = num_threads >> 5;
    int warp_id     = tid >> 5;
    int lane_id     = tid & 31;

    // Shared memory: [0, d) = qi, [d, 2d) = o_acc, [2d, ...) = warp_partial
    extern __shared__ float smem[];
    float* qi            = smem;
    float* o_acc         = smem + d;
    float* warp_partial  = smem + 2 * d;

    float inv_scale = rsqrtf(static_cast<float>(d));

    // Grid-stride over queries.
    for (int q_idx = blockIdx.x; q_idx < M; q_idx += gridDim.x) {

        // 1. Load query row (scalar + UNROLL for MLP).
        const float* Q_row = Q + q_idx * d;
        int step = num_threads * UNROLL;
        int i = tid;
        for (; i + (UNROLL - 1) * num_threads < d; i += step) {
            float regs[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
                regs[u] = Q_row[i + u * num_threads];
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
                qi[i + u * num_threads] = regs[u];
        }
        for (; i < d; i += num_threads)
            qi[i] = Q_row[i];

        // Zero output accumulator.
        for (int k = tid; k < d; k += num_threads)
            o_acc[k] = 0.0f;

        float m = -FLT_MAX;
        float l = 0.0f;

        __syncthreads();

        // 2. Main loop: iterate over all KV pairs.
        for (int j = 0; j < N; j++) {

            // 2a. Dot product: s = Q[i] · K[j]  (UNROLL, float4 from shared qi).
            float dot = 0.0f;
            const float* kj = K + j * d;
            // float4 vectorized dot using qi in shared memory (aligned).
            int n4 = d >> 2;
            const float4* qi4 = reinterpret_cast<const float4*>(qi);
            int k = tid;
            int step4 = num_threads * UNROLL;
            for (; k + (UNROLL - 1) * num_threads < n4; k += step4) {
                float4 kj_regs[UNROLL];
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    int base = (k + u * num_threads) << 2;
                    kj_regs[u] = make_float4(
                        kj[base], kj[base+1], kj[base+2], kj[base+3]);
                }
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    int idx = k + u * num_threads;
                    float4 qv = qi4[idx];
                    dot += qv.x * kj_regs[u].x + qv.y * kj_regs[u].y
                         + qv.z * kj_regs[u].z + qv.w * kj_regs[u].w;
                }
            }
            for (; k < n4; k += num_threads) {
                int base = k << 2;
                float4 qv = qi4[k];
                dot += qv.x * kj[base] + qv.y * kj[base+1]
                     + qv.z * kj[base+2] + qv.w * kj[base+3];
            }
            // Scalar tail.
            for (int t = n4 * 4 + tid; t < d; t += num_threads)
                dot += qi[t] * kj[t];

            // Warp reduce -> warp_partial -> block reduce.
            dot = warp_reduce_sum(dot);
            if (lane_id == 0)
                warp_partial[warp_id] = dot;
            __syncthreads();

            float s = 0.0f;
            if (warp_id == 0) {
                if (lane_id < num_warps)
                    s = warp_partial[lane_id];
                s = warp_reduce_sum(s);
                if (lane_id == 0) {
                    s *= inv_scale;

                    // 2b. Online softmax update.
                    float m_old = m;
                    float m_new = fmaxf(m_old, s);
                    float correction = __expf(m_old - m_new);
                    l = l * correction + __expf(s - m_new);
                    m = m_new;

                    warp_partial[0] = correction;
                    warp_partial[1] = __expf(s - m_new);
                }
            }
            __syncthreads();

            // 2c. Accumulate: o_acc += P_ij * V[j].
            float correction = warp_partial[0];
            float P_ij       = warp_partial[1];
            const float* vj  = V + j * d;

            // float4 in shared memory o_acc (aligned).
            float4* o4 = reinterpret_cast<float4*>(o_acc);
            for (int k2 = tid; k2 < n4; k2 += num_threads) {
                int base = k2 << 2;
                o4[k2].x = o4[k2].x * correction + P_ij * vj[base];
                o4[k2].y = o4[k2].y * correction + P_ij * vj[base+1];
                o4[k2].z = o4[k2].z * correction + P_ij * vj[base+2];
                o4[k2].w = o4[k2].w * correction + P_ij * vj[base+3];
            }
            for (int k2 = n4 * 4 + tid; k2 < d; k2 += num_threads)
                o_acc[k2] = o_acc[k2] * correction + P_ij * vj[k2];
        }

        // 3. Finalize: O[i] = o_acc / l.
        float inv_l = 1.0f / l;
        __syncthreads();

        for (int k = tid; k < d; k += num_threads)
            O[q_idx * d + k] = o_acc[k] * inv_l;

        __syncthreads();
    }
}


extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* O, int M, int N, int d)
{
    if (M <= 0 || N <= 0 || d <= 0)
        return;

    const int threadsPerBlock = 256;
    int num_warps = threadsPerBlock >> 5;

    static int maxBlocks = -1;
    if (maxBlocks < 0) {
        int nb_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nb_per_sm, softmax_attn_kernel<threadsPerBlock>, threadsPerBlock, 0);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        maxBlocks = nb_per_sm * sm_count;
        if (maxBlocks < 1) maxBlocks = 1;
    }

    int blocksPerGrid = std::min(M, maxBlocks);

    int shared_bytes = 2 * d * sizeof(float)
                     + num_warps * sizeof(float);

    softmax_attn_kernel<threadsPerBlock>
        <<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(Q, K, V, O, M, N, d);
    cudaDeviceSynchronize();
}
