#include <cuda_runtime.h>

#define IS_ALIGNED(ptr) (((uintptr_t)(ptr)) & 15 == 0)
#define FETCH_FLOAT4C(ptr, idx) (*reinterpret_cast<const float4*>(&ptr[idx]))
#define FETCH_FLOAT4(ptr, idx) (*reinterpret_cast<float4*>(&ptr[idx]))

template <int KX, int KY, int KZ>
__constant__ float c_kernel[1000]; // assumes KX*KY*KZ <= 1000

template<int TX, int TY, int TZ, int KX, int KY, int KZ>
__global__ void _3d_conv_kernel(
	const float* __restrict__ input,
	float* __restrict__ output,
	int input_depth, int input_rows, int input_cols,
	int output_depth, int output_rows, int output_cols
) {
	constexpr int OUT_X = TX * 4;
	constexpr int SEME_X = ((OUT_X + KX - 1 + 3) / 4) * 4;
	constexpr int SEME_Y = ((TY + KY - 1 + 3) / 4) * 4;
	constexpr int SEME_Z = ((TZ + KZ - 1 + 3) / 4) * 4;
	
	__shared__ float smem[SEME_Z][SEME_Y][SEME_X];

	const int out_x0 = blockIdx.x * OUT_X;
	const int out_y0 = blockIdx.y * TY;
	const int out_z0 = blockIdx.z * TZ;

	const int tid = threadIdx.z * (TY * TX) + threadIdx.y * TX + threadIdx.x;
	const int nthreads = TX * TY * TZ;

	constexpr int SMEM
}

extern "C" void solve(
	const float* input, const float* kernel, float* output, int input_depth,
	int input_rows, int input_cols, int kernel_depth, int kernel_rows,
	int kernel_cols
) {

}