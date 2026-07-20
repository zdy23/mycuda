#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float *A, float *B, float *C, int N) {
  // Kernel code here
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++) {
    if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("Results do not match at index %d: host %f, gpu %f\n", i,
             hostRef[i], gpuRef[i]);
      return;
    }
  }
  printf("Results match!\n");
}

int main() {
  int N = 1024;
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  float *h_C_ref = (float *)malloc(size);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i);
    h_C_ref[i] = h_A[i] + h_B[i];
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define grid and block structures
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Launch kernel
  kernel<<<grid, block>>>(d_A, d_B, d_C, N);

  // Copy result from device to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Check results
  checkResult(h_C_ref, h_C, N);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}
