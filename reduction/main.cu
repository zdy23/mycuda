#include <cuda_runtime.h>

#define UNROLL 8

__device__ float warp_reduce(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;
}

__device__ float block_reduce(float val, float* smem) {
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int warps = blockDim.x >> 5;

	// Each warp performs partial reduction
	val = warp_reduce(val);

	// Write reduced value to shared memory
	if (lane == 0) smem[warp] = val;

	__syncthreads();

	if (warp == 0) {
		val = (lane < warps) ? smem[lane] : 0.0f;
		val = warp_reduce(val);
		if (lane == 0) smem[0] = val;
	}
	__syncthreads();
	return smem[0];
}

__global__ void reduceKernel_small(const float* input, float* output, int N) {
	extern __shared__ float smem[];
	
	const int stride = blockDim.x * UNROLL;
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int warps = blockDim.x >> 5;

	float sum = 0.0f;
	float *sums = smem;

	// thread coarsening
	int i = tid;
	for (; i + stride - blockDim.x < N; i += stride) {
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			sum += input[i + u * blockDim.x];
		}
	}

	for (; i < N; i += blockDim.x) {
		sum += input[i];
	}

	sum = warp_reduce(sum);

	if (lane == 0) sums[warp] = sum;
	__syncthreads();

	if (tid == 0) {
		float val = sums[0];
		for (int w = 1; w < warps; w++) {
			val += sums[w];
		}
		output[0] = val;
	}
	__syncthreads();

}

__global__ void reduceKernel_large(const float* input, float* partial, int N) {
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int stride = blockDim.x * UNROLL;

	float sum = 0.0f;

	// grid-stride loop + unrolling
	int i = blockIdx.x * stride + tid;
	for (; i + stride - blockDim.x < N; i += stride * gridDim.x) {
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			sum += input[i + u * blockDim.x];
		}
	}

	// tail case
	for (; i < N; i += blockDim.x) {
		sum += input[i];
	}

	sum = block_reduce(sum, smem);
	if (tid == 0) partial[blockIdx.x] = sum;
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
	if (N <= 0) return;

	const int threads = 256;
	const int warps = threads / 32;
	const size_t smem_size = warps * sizeof(float);

	if (N <= threads * UNROLL * 8) {
		// small: one block sweeps everything, writes output[0]
		reduceKernel_small<<<1, threads, smem_size>>>(input, output, N);
		cudaDeviceSynchronize();
		return;
	}

	// large: two-pass block-tile reduction
	int blocks = (N + threads * UNROLL - 1) / (threads * UNROLL);
	if (blocks > 1024) blocks = 1024;

	float *partial = nullptr;
	cudaMalloc(&partial, blocks * sizeof(float));

	reduceKernel_large<<<blocks, threads, smem_size>>>(input, partial, N);
	reduceKernel_large<<<1, threads, smem_size>>>(partial, output, blocks);

	cudaFree(partial);
	cudaDeviceSynchronize();
}