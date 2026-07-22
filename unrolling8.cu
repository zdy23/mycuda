#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n) {
  constexpr unsigned int UNROLL_FACTOR = 8;
  __shared__ int sdata[iBlockSize];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * (blockIdx.x * UNROLL_FACTOR) + tid;

  int sum = 0;

  if (idx + 7 * blockDim.x < n) {
    sum += g_idata[idx];
    sum += g_idata[idx + blockDim.x];
    sum += g_idata[idx + blockDim.x * 2];
    sum += g_idata[idx + blockDim.x * 3];
    sum += g_idata[idx + blockDim.x * 4];
    sum += g_idata[idx + blockDim.x * 5];
    sum += g_idata[idx + blockDim.x * 6];
    sum += g_idata[idx + blockDim.x * 7];
  } else {
    for (unsigned int i = 0; i < UNROLL_FACTOR && idx + i * blockDim.x < n;
         i++) {
      sum += g_idata[idx + i * blockDim.x];
    }
  }
  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    int val = sdata[tid];
    for (int offset = 32; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0)
      g_odata[blockIdx.x] = val;
  }
}

void gpuReduce(int *d_idata, int *d_odata, unsigned int n,
               unsigned int blockSize) {
  unsigned int gridSize = (n + (unsigned long long)blockSize * 8 - 1) /
                          ((unsigned long long)blockSize * 8);

  reduceCompleteUnroll<256><<<gridSize, blockSize>>>(d_idata, d_odata, n);
  cudaDeviceSynchronize();
  if (gridSize > 1) {
    reduceCompleteUnroll<256><<<1, blockSize>>>(d_odata, d_odata, gridSize);
    cudaDeviceSynchronize();
  }
}

int cpuReduce(int *data, unsigned int size) {
  int sum = 0;
  for (unsigned int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum;
}

int main() {
  unsigned int n = 1 << 20; // 1M elements
  size_t bytes = n * sizeof(int);

  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(1024 * sizeof(int));

  srand(time(NULL));
  for (unsigned int i = 0; i < n; i++) {
    h_idata[i] = 1;
  }

  int *d_idata = nullptr;
  int *d_odata = nullptr;

  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, 1024 * sizeof(int));

  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

  const int blockSize = 256;
  int gridSize = (n + blockSize * 8 - 1) / (blockSize * 8);

  printf("Array size: %d\n", n);
  printf("Grid size: %d, Block size: %d\n", gridSize, blockSize);

  int *gpu_sum = (int *)malloc(sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  gpuReduce(d_idata, d_odata, n, blockSize);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemcpy(gpu_sum, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

  int cpu_sum = cpuReduce(h_idata, n);
  printf("CPU sum: %d, GPU sum: %d\n", cpu_sum, *gpu_sum);
  printf("Match: %s\n", (cpu_sum == *gpu_sum) ? "Yes" : "No");

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time: %f ms\n", milliseconds);

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceReset();

  return 0;
}
