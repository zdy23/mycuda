#include <stdio.h>

#define N 8

__global__ void interleavedReduce(int *input, int *output) {
  __shared__ int data[N];

  int tid = threadIdx.x;

  // 数据加载到共享内存
  data[tid] = input[tid];

  __syncthreads();

  // 交错归约
  for (int stride = 1; stride < N; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      data[tid] += data[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    *output = data[0];
}

int main() {
  int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};

  int *d_input;
  int *d_output;

  cudaMalloc(&d_input, sizeof(int) * N);
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

  interleavedReduce<<<1, N>>>(d_input, d_output);

  int result;

  cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  printf("sum=%d\n", result);

  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
