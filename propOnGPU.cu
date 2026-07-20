#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  // 先检查系统有多少 CUDA 设备
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  if (err != cudaSuccess) {
    printf("Failed to get device count: %s\n", cudaGetErrorString(err));
    return -1;
  }

  if (deviceCount == 0) {
    printf("No CUDA-capable device found!\n");
    return -1;
  }

  printf("Found %d CUDA device(s)\n", deviceCount);

  for (int device = 0; device < deviceCount; device++) {
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
      printf("Error getting properties for device %d: %s\n", device,
             cudaGetErrorString(err));
      continue;
    }

    printf("\n=== Device %d: %s ===\n", device, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
  }

  return 0;
}
