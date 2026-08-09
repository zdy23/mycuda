#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

/*
 Q:  M x d
 K:  N x d
 V:  N x d
 O:  M x d

 Fused pipeline:
   score = Q @ K^T / sqrt(d)
   online row softmax
   O = softmax(score) @ V
*/

#define LOG2_E 1.4426950408889634f

template <int TPB>
__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int d, float inv_scale
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    // logits, reduction workspace, and output accumulator share dynamic storage.
    extern __shared__ float smem[];
    float* logits = smem;
    float* reduce = logits + TPB;
    float* accum = reduce + TPB;
    const float* q = Q + (size_t)row * d;

    for (int col = tid; col < d; col += TPB)
        accum[col] = 0.0f;
    __syncthreads();

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    for (int tile_start = 0; tile_start < N; tile_start += TPB) {
        int key = tile_start + tid;
        float score = -FLT_MAX;
        if (key < N) {
            score = 0.0f;
            for (int x = 0; x < d; x++)
                score += q[x] * K[(size_t)key * d + x];
            score *= inv_scale;
        }
        logits[tid] = score;
        reduce[tid] = score;
        __syncthreads();

        for (int stride = TPB / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        float tile_max = reduce[0];

        reduce[tid] = (key < N)
            ? exp2f((score - tile_max) * LOG2_E)
            : 0.0f;
        __syncthreads();
        for (int stride = TPB / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        float tile_sum = reduce[0];

        float new_max = fmaxf(row_max, tile_max);
        float old_scale = (row_max == -FLT_MAX)
            ? 0.0f : exp2f((row_max - new_max) * LOG2_E);
        float tile_scale = exp2f((tile_max - new_max) * LOG2_E);

        for (int col = tid; col < d; col += TPB) {
            float value = accum[col] * old_scale;
            for (int k = 0; k < TPB && tile_start + k < N; k++) {
                float weight = exp2f((logits[k] - tile_max) * LOG2_E);
                value += weight * tile_scale *
                    V[(size_t)(tile_start + k) * d + col];
            }
            accum[col] = value;
        }

        row_sum = row_sum * old_scale + tile_sum * tile_scale;
        row_max = new_max;
        __syncthreads();
    }

    for (int col = tid; col < d; col += TPB)
        output[(size_t)row * d + col] = accum[col] / row_sum;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
    float* output, int M, int N, int d
) {
    constexpr int TPB = 256;
    size_t shared_bytes = (size_t)(2 * TPB + d) * sizeof(float);
    float inv_scale = 1.0f / sqrtf((float)d);

    fused_attention_kernel<TPB><<<M, TPB, shared_bytes>>>(
        Q, K, V, output, M, N, d, inv_scale);
}
