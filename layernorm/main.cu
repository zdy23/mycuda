#include <cuda_runtime.h>

__global__ void layernorm_kernel(const float *input, const float *gamma, const float *beta,
	float *output, int N, int C, float eps)
{
	
}


// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps)
{

}
