#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void sumMatrix(float *a, float *b, float *c, int nx, int ny) {
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int tid = iy * nx + ix;
  if (ix < nx && iy < ny) {
    c[tid] = a[tid] + b[tid];
  }
}

// 初始化矩阵
void initialData(float *ip, int size) {
  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

// 执行 + 计时：block 由调用方传入，grid 在函数内计算
float runSumMatrix(float *d_a, float *d_b, float *d_c, int nx, int ny,
                   dim3 block) {
  if (block.x * block.y * block.z > 1024) {
    printf("skip block(%u,%u): threads/block=%u > 1024\n", block.x, block.y,
           block.x * block.y * block.z);
    return -1.f;
  }

  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // 预热（不计时）
  sumMatrix<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
  cudaDeviceSynchronize();

  float elapsedTime = 0.f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  sumMatrix<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("sumMatrix <<<grid(%u,%u), block(%u,%u)>>>  %.3f ms\n", grid.x, grid.y,
         block.x, block.y, elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsedTime;
}

int main() {
  int nx = 1 << 12, ny = 1 << 12;
  int nxy = nx * ny;
  size_t nBytes = nxy * sizeof(float);

  printf("Matrix size: nx=%d ny=%d\n", nx, ny);

  // host
  float *h_a = (float *)malloc(nBytes);
  float *h_b = (float *)malloc(nBytes);
  float *h_c = (float *)malloc(nBytes);

  initialData(h_a, nxy);
  initialData(h_b, nxy);

  // device
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, nBytes);
  cudaMalloc((void **)&d_b, nBytes);
  cudaMalloc((void **)&d_c, nBytes);

  cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
  
  int configs[][2] = {
      {32, 32}, {16, 16}, {8, 8}, {4, 4}, {2, 2}, {1, 1},
  };
  int nConfigs = (int)(sizeof(configs) / sizeof(configs[0]));

  for (int i = 0; i < nConfigs; i++) {
    dim3 block(configs[i][0], configs[i][1]);
    runSumMatrix(d_a, d_b, d_c, nx, ny, block);
  }

  // 抽查最后一次结果（1+1 随机，只打印几个元素看是否合理）
  cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);
  printf("check: h_c[0]=%.2f (a+b=%.2f), h_c[1]=%.2f (a+b=%.2f)\n", h_c[0],
         h_a[0] + h_b[0], h_c[1], h_a[1] + h_b[1]);

  // 清理内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  cudaDeviceReset();
  return 0;
}
