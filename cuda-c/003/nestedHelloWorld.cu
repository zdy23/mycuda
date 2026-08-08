#include <cuda_runtime.h>
#include <stdio.h>

__global__ void nestedHelloWorld(const int iSize, int iDepth) {
  int tid = threadIdx.x;
  printf("Recursion %d, Hello World from block %d and thread %d\n", iDepth, tid,
         blockIdx.x);

  // condition to stop recursion
  if (iSize <= 1)
    return;

  // reduce block size to half
  int nthreads = iSize / 2;

  // launch kernel recursively
  if (tid == 0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, iDepth + 1);
    printf("===nested execution depth: %d===\n", iDepth);
  }
}

int main() {
  int nthreads = 32;
  nestedHelloWorld<<<1, nthreads>>>(nthreads, 0);
  cudaDeviceSynchronize();
  return 0;
}
