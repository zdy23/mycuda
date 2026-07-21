#include <cuda_runtime.h>
#include <random>
#include <stdio.h>

void initializeData(int *data, int n) {
  std::random_device rd;  // 随机数生成器
  std::mt19937 gen(rd()); // 使用 Mersenne Twister 算法的随机数引擎
  std::uniform_int_distribution<> dis(1, 100); // 生成 1 到 100 之间的随机整数

  for (int i = 0; i < n; ++i) {
    data[i] = dis(gen); // 生成随机整数并赋值给数组
  }
}

__global__ void reduceDivergent(int *input, int *output, int n) {
  __shared__ int sdata[256]; // Shared memory for partial sums

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0; // Load input into shared memory
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void reduceLessDivergent(int *input, int *output, int n) {
  __shared__ int sdata[256]; // Shared memory for partial sums

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0;
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void reduceInterleaved(int *input, int *output, int n) {
  __shared__ int sdata[256]; // Shared memory for partial sums

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

// Shared Memory + Warp Unrolling
__global__ void reduceWarpUnrolling(int *input, int *output, int n) {
  __shared__ int sdata[256];

  unsigned int tid = threadIdx.x;
  // each block covers 2 * blockDim elements
  unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;

  sdata[tid] = (idx < n) ? input[idx] : 0;
  if (idx + blockDim.x < n) {
    sdata[tid] += input[idx + blockDim.x];
  }
  __syncthreads();

  // reduce in shared until 64 partials left (sdata[0..63])
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // last warp: fold 64 -> 32, then shuffle 32 -> 1
  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    int val = sdata[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0) {
      output[blockIdx.x] = val;
    }
  }
}

int main() {
  int n = 1 << 14;
  int *h_input = (int *)malloc(n * sizeof(int));
  int *h_output = (int *)malloc((n / 256) * sizeof(int));

  initializeData(h_input, n);

  int *d_input, *d_output;
  cudaMalloc((void **)&d_input, n * sizeof(int));
  cudaMalloc((void **)&d_output, (n / 256) * sizeof(int));
  cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  reduceDivergent<<<n / 256, 256>>>(d_input, d_output, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecondsDivergent = 0;
  cudaEventElapsedTime(&millisecondsDivergent, start, stop);

  printf("Divergent reduce time: %f ms\n", millisecondsDivergent);

  cudaEventRecord(start);
  reduceLessDivergent<<<n / 256, 256>>>(d_input, d_output, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecondsLessDivergent = 0;
  cudaEventElapsedTime(&millisecondsLessDivergent, start, stop);

  printf("Less divergent reduce time: %f ms\n", millisecondsLessDivergent);

  cudaEventRecord(start);
  reduceInterleaved<<<n / 256, 256>>>(d_input, d_output, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecondsInterleaved = 0;
  cudaEventElapsedTime(&millisecondsInterleaved, start, stop);

  printf("Interleaved reduce time: %f ms\n", millisecondsInterleaved);

  cudaEventRecord(start);
  // 2 loads/thread => half as many blocks
  reduceWarpUnrolling<<<n / (256 * 2), 256>>>(d_input, d_output, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecondsWarpUnrolling = 0;
  cudaEventElapsedTime(&millisecondsWarpUnrolling, start, stop);

  printf("Warp unrolling reduce time: %f ms\n", millisecondsWarpUnrolling);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_input);
  free(h_output);

  return 0;
}
