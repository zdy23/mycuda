#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,
                                   unsigned int isize) {

  // 线程索引 + 块索引
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;

  // 指向当前线程块的输入数据和输出数据
  int *idata = g_idata + bid * isize;
  int *odata = g_odata + bid;

  // 最后一次求和
  // 最后一个线程块的线程数为2时，直接计算结果
  if (isize == 2 && tid == 0) {
    *odata = idata[0] + idata[1];
    return;
  }

  // 每个线程块内的线程数为isize的一半
  int istride = isize >> 1;

  // 每个线程块内的线程进行归约操作
  if (tid < istride) {
    idata[tid] += idata[tid + istride];
  }

  // 递归调用，继续归约操作
  if (tid == 0 && istride > 1) {
    gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
  }
}

int main() {

  int n = 1 << 8;
  int bytes = n * sizeof(int);

  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(sizeof(int));

  for (int i = 0; i < n; i++) {
    h_idata[i] = 1;
  }

  int *d_idata, *d_odata;
  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, sizeof(int));

  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  gpuRecursiveReduce<<<gridSize, blockSize>>>(d_idata, d_odata, n);

  cudaMemcpy(h_odata, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Sum: %d\n", h_odata[0]);

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
  cudaDeviceReset();

  return 0;
}
