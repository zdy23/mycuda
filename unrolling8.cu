#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduceUnrollWarps8(int *input, int *output, unsigned int n) {
  __shared__ int sdata[256]; // Shared memory for partial sums
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  int sum = 0;
  if (idx < n)
    sum += input[idx];
  if (idx + blockDim.x < n)
    sum += input[idx + blockDim.x];
  if (idx + 2 * blockDim.x < n)
    sum += input[idx + 2 * blockDim.x];
  if (idx + 3 * blockDim.x < n)
    sum += input[idx + 3 * blockDim.x];
  if (idx + 4 * blockDim.x < n)
    sum += input[idx + 4 * blockDim.x];
  if (idx + 5 * blockDim.x < n)
    sum += input[idx + 5 * blockDim.x];
  if (idx + 6 * blockDim.x < n)
    sum += input[idx + 6 * blockDim.x];
  if (idx + 7 * blockDim.x < n)
    sum += input[idx + 7 * blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile int *vmem = sdata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void reduceCompleteUnrollingWarp8(int *input, int *output,
                                             unsigned int n) {
  __shared__ int sdata[256]; // Shared memory for partial sums
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  int sum = 0;
  if (idx < n)
    sum += input[idx];
  if (idx + blockDim.x < n)
    sum += input[idx + blockDim.x];
  if (idx + 2 * blockDim.x < n)
    sum += input[idx + 2 * blockDim.x];
  if (idx + 3 * blockDim.x < n)
    sum += input[idx + 3 * blockDim.x];
  if (idx + 4 * blockDim.x < n)
    sum += input[idx + 4 * blockDim.x];
  if (idx + 5 * blockDim.x < n)
    sum += input[idx + 5 * blockDim.x];
  if (idx + 6 * blockDim.x < n)
    sum += input[idx + 6 * blockDim.x];
  if (idx + 7 * blockDim.x < n)
    sum += input[idx + 7 * blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  // in-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = sdata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main() {

  int N = 1 << 20; // Size of the array (1 million elements)
  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output = (int *)malloc(sizeof(int)); // Output will be a single value
  int *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(int));
  cudaMalloc((void **)&d_output, N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_input[i] = 1; // For simplicity, fill the array with 1s
  }
  int blockSize = 256;
  int gridSize = (N + blockSize * 8 - 1) / (blockSize * 8);

  cudaFree(0); // Initialize CUDA context
  for (int i = 0; i < 5; i++) {
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    reduceUnrollWarps8<<<gridSize, blockSize>>>(d_input, d_output, N);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  reduceUnrollWarps8<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  float reduceUnrolling8_milliseconds = 0;
  cudaEventElapsedTime(&reduceUnrolling8_milliseconds, start, stop);
  printf("Result of reduction: %d\n", h_output[0]);
  printf("Time taken for reduction with warp unrolling: %f ms\n",
         reduceUnrolling8_milliseconds);

  cudaEventRecord(start);
  reduceCompleteUnrollingWarp8<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  float reduceCompleteUnrollingWarp8_milliseconds = 0;
  cudaEventElapsedTime(&reduceCompleteUnrollingWarp8_milliseconds, start, stop);
  printf("Result of reduction: %d\n", h_output[0]);
  printf("Time taken for complete unrolling with warp unrolling: %f ms\n",
         reduceCompleteUnrollingWarp8_milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);
  cudaDeviceReset();

  return 0;
}
