#include <cuda_runtime.h>
#include <stdio.h>

// 深度3
__global__ void depth3_kernel() {
  printf("  [Depth 3] Hello from the deepest!\n");
}

// 深度2
__global__ void depth2_kernel() {
  printf("  [Depth 2] Hello from middle\n");
  // 使用 TailLaunch 启动深度3，不增加嵌套深度
  depth3_kernel<<<1, 1, 0, cudaStreamTailLaunch>>>();
  printf("  [Depth 2] Launched depth3, returning immediately\n");
}

// 深度1
__global__ void depth1_kernel() {
  printf("[Depth 1] Hello from parent\n");
  // 使用 TailLaunch 启动深度2
  depth2_kernel<<<1, 1, 0, cudaStreamTailLaunch>>>();
  printf("[Depth 1] Launched depth2, returning immediately\n");
}

int main() {
  printf("Launching parent kernel with TailLaunch...\n\n");

  // 启动父内核
  depth1_kernel<<<1, 1>>>();

  // CPU等待所有GPU任务完成
  cudaDeviceSynchronize();
  printf("\nAll done!\n");

  return 0;
}
