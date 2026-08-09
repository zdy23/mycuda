#include <cuda_runtime.h>
#include <cstdio>

int main() {
  int n = 0;
  cudaGetDeviceCount(&n);
  for (int i = 0; i < n; i++) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, i);
    printf("GPU %d: %s\n", i, p.name);
    printf("  SMs: %d\n", p.multiProcessorCount);
    printf("  Max threads/block: %d\n", p.maxThreadsPerBlock);
    printf("  Max threads/SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Warp size: %d\n", p.warpSize);
    printf("  Global mem: %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("  Shared mem/block: %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("  CC: %d.%d\n", p.major, p.minor);
  }
}