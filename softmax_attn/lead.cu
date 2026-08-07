#include <cuda_runtime.h>

// 定义常量 TILE_WIDTH 和 COARSE_FACTOR
// 分别表示 每个线程块的宽度和粗粒度因子，用于矩阵乘法和转置操作
#define TILE_WIDTH 16
#define COARSE_FACTOR 1
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K, const float scale) {
    
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int colStart = COARSE_FACTOR * TILE_WIDTH * blockIdx.x + threadIdx.x;

    float sum[COARSE_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }

    for (int phase = 0; phase < (N + TILE_WIDTH - 1) / TILE_WIDTH; phase++) {
        // Collaboratively load A into shared memory
        As[threadIdx.y][threadIdx.x] = 
            (row < M && (phase * TILE_WIDTH + threadIdx.x) < N) ? 
            A[row * N + phase * TILE_WIDTH + threadIdx.x] : 0.0f;

        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            // Collaboratively load B into shared memory
            Bs[threadIdx.y][threadIdx.x] = 
                ((phase * TILE_WIDTH + threadIdx.y) < N && col < K) ?
                B[(phase * TILE_WIDTH + threadIdx.y) * K + col] : 0.0f;
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH; j++) {
                sum[c] += As[threadIdx.y][j] * Bs[j][threadIdx.x];
            }
            __syncthreads();
        }
        
    }

    // Scale and sum
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        if (row < M && col < K) {
            C[row * K + col] = sum[c] / scale;
        }
    }
    
}

#define TPB 256

__global__ void row_softmax_kernel(float* input, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row < M) {

        __shared__ float s_max[TPB];
        __shared__ float s_sum[TPB];

        float max_val = -INFINITY;
        float sum_val = 0.0f;

        #pragma unroll
        for (int j = tid; j < N; j += TPB) {
            float val = input[row * N + j];
            if (val > max_val) {
                sum_val = sum_val * expf(max_val - val) + 1.0f;
                max_val = val;
            } else {
                sum_val += expf(val - max_val);
            }
        }

        s_max[tid] = max_val;
        s_sum[tid] = sum_val;
        __syncthreads();


        #pragma unroll
        for (int stride = TPB / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float my_max = s_max[tid];
                float neighbor_max = s_max[tid + stride];
                float my_sum = s_sum[tid];
                float neighbor_sum = s_sum[tid + stride];

                if (neighbor_max > -INFINITY) {
                    if (neighbor_max > my_max) {
                        s_sum[tid] = my_sum * expf(my_max - neighbor_max) + neighbor_sum;
                        s_max[tid] = neighbor_max;
                    } else {
                        s_sum[tid] = neighbor_sum * expf(neighbor_max - my_max) + my_sum;
                    }
                }
            }

            __syncthreads();
        }

        float row_max = s_max[0];
        float row_sum = s_sum[0];

        // Normalize
        for (int j = tid; j < N; j += TPB) {
            float score = input[row * N + j];
            input[row * N + j] = expf(score - row_max) / row_sum;
        }
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < cols && r < rows) {
        output[c  * rows + r] = input[r * cols + c];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {

    // Allocate device memory
    float* d_score;
    float* d_KT;
    cudaMalloc(&d_score, M * N * sizeof(float));
    cudaMalloc(&d_KT, sizeof(float) * d * N);

    dim3 block(TILE_WIDTH, TILE_WIDTH);

    // Step 1: K^T = transpose(K) -> Shape: d x N
    dim3 grid_trans((d + TILE_WIDTH - 1) / TILE_WIDTH, 
                    (N + TILE_WIDTH - 1) / TILE_WIDTH);
    transpose_kernel<<<grid_trans, block>>>(K, d_KT, N, d);

    // Step 2: score = QK^T / sqrt(d) -> Shape: M x N
    dim3 grid_score((N + (TILE_WIDTH * COARSE_FACTOR) - 1) / (TILE_WIDTH * COARSE_FACTOR), 
                    (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matrix_multiplication_kernel<<<grid_score, block>>>(Q, d_KT, d_score, M, d, N, sqrt(d));

    // Step 3: row_softmax(scaled_score) (in-place modification)
    row_softmax_kernel<<<M, TPB>>>(d_score, M, N);

    // Step 4: output = score * V -> Shape: M x d
    dim3 grid_out((d + (TILE_WIDTH * COARSE_FACTOR) - 1) / (TILE_WIDTH * COARSE_FACTOR), 
                  (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_multiplication_kernel<<<grid_out, block>>>(d_score, V, output, M, N, d, 1.0f);

    cudaFree(d_score);
    cudaFree(d_KT);
}
