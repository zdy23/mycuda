#include <cuda_runtime.h>

__global__ void vector_add(const float4* A, const float4* B, float* C, int N, int N_4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int base = idx * 4;
	if (idx < N_4) {
		float4 a = A[idx];
		float4 b = B[idx];
		if (base + 0 < N) C[base + 0] = a.x + b.x;
		if (base + 1 < N) C[base + 1] = a.y + b.y;
		if (base + 2 < N) C[base + 2] = a.z + b.z;
		if (base + 3 < N) C[base + 3] = a.w + b.w;
	}
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
	int N_4 = (N + 3) / 4; // Number of float4 elements (rounded up)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_4 + threadsPerBlock - 1) / threadsPerBlock;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N, N_4);
    cudaDeviceSynchronize();
}
