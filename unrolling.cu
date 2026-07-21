#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduceWarpUnrolling_1(int *input, int *output, int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

  int *data = input + blockDim.x * 2 + blockIdx.x;
  if (idx + blockDim.x < n) {
    data[tid] += data[tid + blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0)
    output[blockIdx.x] = data[0];
}

__global__ void reduceWarpUnrolling_2(int *input, int *output, int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

  int *data = input + blockDim.x * 2 + blockIdx.x;
  if (idx + blockDim.x < n) {
    data[tid] += data[tid + blockDim.x];
    data[tid] += data[tid + 2 * blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0)
    output[blockIdx.x] = data[0];
}

int main() {

  int N = 1 << 20; // Size of the array (1 million elements)
  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output = (int *)malloc(sizeof(int)); // Output will be a single value
  int *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(int));
  cudaMalloc((void **)&d_output, sizeof(int));
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Initialize input array
  for (int i = 0; i < N; i++) {
    h_input[i] = 1; // For simplicity, fill the array with 1s
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  reduceWarpUnrolling_1<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds1 = 0;
  cudaEventElapsedTime(&milliseconds1, start, stop);
  printf("Time taken for reduceWarpUnrolling_1: %f ms\n", milliseconds1);

  cudaEventRecord(start);
  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  reduceWarpUnrolling_2<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds2 = 0;
  cudaEventElapsedTime(&milliseconds2, start, stop);
  printf("Time taken for reduceWarpUnrolling_2: %f ms\n", milliseconds2);

  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  cudaDeviceReset();

  return 0;
}
