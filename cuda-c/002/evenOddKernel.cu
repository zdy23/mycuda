#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void kernelDivergent(int *d_C, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    if (tid % 2 == 0) {
      d_C[tid] = tid * 2;
    } else {
      d_C[tid] = tid * 3 + 1;
    }
  }
}

__global__ void kernelNotDivergent(int *d_C, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int condition = tid * 2;
    d_C[tid] = condition ? (tid * 3 + 1) : (tid * 2);
  }
}

int main(int argc, char **argv) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s using Device %d\n", deviceProp.name, dev);

  // set up data size
  int size = 64;
  int blockSize = 64;
  if (argc > 1)
    blockSize = atoi(argv[1]);
  if (argc > 2)
    size = atoi(argv[2]);
  printf("Data size: %d, Block size: %d\n", size, blockSize);

  // set up d_C
  int *d_C;
  size_t nBytes = size * sizeof(int);
  cudaMalloc((int **)&d_C, nBytes);

  // alloc gpu memory
  int *h_C;
  h_C = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++) {
    h_C[i] = i;
  }

  cudaMemcpy(d_C, h_C, size * sizeof(int), cudaMemcpyHostToDevice);

  // set up exe configuration
  dim3 block(blockSize, 1);
  dim3 grid((size + blockSize - 1) / blockSize, 1);
  printf("Grid size: %d, Block size: %d\n", grid.x, block.x);

  // run warmup kernel to remove overhead
  double iStart, iElaps;
  cudaDeviceSynchronize();

  // run kernel Divergent
  iStart = seconds();
  kernelDivergent<<<grid, block>>>(d_C, size);
  iElaps = seconds();
  printf("Divergent kernel execution time: %f sec\n", iElaps - iStart);

  // run kernel NotDivergent
  iStart = seconds();
  kernelNotDivergent<<<grid, block>>>(d_C, size);
  iElaps = seconds();
  printf("Not Divergent kernel execution time: %f sec\n", iElaps - iStart);

  cudaFree(d_C);
  free(h_C);

  return 0;
}
