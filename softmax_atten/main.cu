#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d)
{
	
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d)
{

}