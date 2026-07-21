#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

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

int cpuReduce(int *data, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += data[i];
  }
  return sum;
}

int main() {

  const unsigned int N = 1 << 24;
  const int blockSize = 256;
  const int gridSize = (N + blockSize * 8 - 1) / (blockSize * 8);

  printf("Grid size: %d, Block size: %d\n", gridSize, blockSize);

  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output = (int *)malloc(gridSize * sizeof(int));
  int *h_partialSums = (int *)malloc(gridSize * sizeof(int));

  for (int i = 0; i < N; i++) {
    h_input[i] = 1; // Initialize input array with 1s
  }

  int *d_input, *d_partialSums, *d_output;
  CHECK(cudaMalloc((void **)&d_input, N * sizeof(int)));
  CHECK(cudaMalloc((void **)&d_partialSums, gridSize * sizeof(int)));
  CHECK(cudaMalloc((void **)&d_output, gridSize * sizeof(int)));

  CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  // === test kernel1 ===
  CHECK(cudaEventRecord(start));
  reduceCompleteUnrollingWarp8<<<gridSize, blockSize>>>(d_input, d_partialSums,
                                                        N);
  reduceUnrollWarps8<<<1, blockSize>>>(d_partialSums, d_output, N);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms1 = 0;
  CHECK(cudaEventElapsedTime(&ms1, start, stop));
  CHECK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

  int cpu_result = cpuReduce(h_input, N);
  printf("CPU result: %d, GPU result: %d\n", cpu_result, h_output[0]);
  printf("Time: %.3f ms\n", ms1);
  printf(h_output[0] == cpu_result ? "Success\n" : "Failure\n");

  // === test kernel2 ===
  CHECK(cudaEventRecord(start));
  reduceUnrollWarps8<<<gridSize, blockSize>>>(d_input, d_partialSums, N);
  reduceUnrollWarps8<<<1, blockSize>>>(d_partialSums, d_output, N);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms2 = 0;
  CHECK(cudaEventElapsedTime(&ms2, start, stop));
  CHECK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

  printf("CPU result: %d, GPU result: %d\n", cpu_result, h_output[0]);
  printf("Time: %.3f ms\n", ms2);
  printf(h_output[0] == cpu_result ? "Success\n" : "Failure\n");

  return 0;
}
