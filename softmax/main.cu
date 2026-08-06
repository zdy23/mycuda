#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>

namespace cg = cooperative_groups;

#define UNROLL 8

#ifdef __clang__
#define LDCS(ptr) (*(ptr))
#else
#define LDCS(ptr) __ldcs(ptr)
#endif

__device__ float warp_reduce_max(float val) {
	for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
		val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
	}
	return val;
}

__device__ float warp_reduce_sum(float val) {
	for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;
}

// block-level reduction for max and sum using shared memory
// v is the value to reduce, smem is a pointer to shared memory
__device__ float block_reduce_max(float v, float* smem) {
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int warps = blockDim.x >> 5;

	// reduce the values within each warp
	v = warp_reduce_max(v);
	if (lane == 0) smem[warp] = v;
	__syncthreads();

	// 1 block = 8 warps = 256 threads
	if (warp == 0) {
		// lane 0~7 will reduce the 8 values in smem[0~7]
		v = (lane < warps) ? smem[lane] : -INFINITY;
		// reduce the 8 values in smem[0~7] to smem[0]
		v = warp_reduce_max(v);
		if (lane == 0) smem[0] = v;
	}
	__syncthreads();
	return smem[0];
}

__device__ float block_reduce_sum(float v, float* smem) {
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int warps = blockDim.x >> 5;

	// reduce the values within each warp
	v = warp_reduce_sum(v);
	if (lane == 0) smem[warp] = v;
	__syncthreads();

	if (warp == 0) {
		v = (lane < warps) ? smem[lane] : 0.f;
		v = warp_reduce_sum(v);
		if (lane == 0) smem[0] = v;
	}
	__syncthreads();
	return smem[0];
}

__global__ void softmax_multi_fused(const float* x, float* y,
									float* pmax, float* psum, int N) {
	cg::grid_group grid = cg::this_grid();
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int stride = blockDim.x * UNROLL;

	float m = -INFINITY;
	int i = blockIdx.x * stride + tid;
	for (; i + stride - blockDim.x < N; i += stride * gridDim.x) {
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			m = fmaxf(m, x[i + u * blockDim.x]);
	}
	for (; i < N; i += blockDim.x)
		m = fmaxf(m, x[i]);
	m = block_reduce_max(m, smem);
	if (tid == 0) pmax[blockIdx.x] = m;
	grid.sync();

	m = -INFINITY;
	for (int j = tid; j < gridDim.x; j += blockDim.x)
		m = fmaxf(m, pmax[j]);
	m = block_reduce_max(m, smem);

	float s = 0.f;
	i = blockIdx.x * stride + tid;
	for (; i + stride - blockDim.x < N; i += stride * gridDim.x) {
		float reg[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			reg[u] = LDCS(&x[i + u * blockDim.x]);
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			float e = expf(reg[u] - m);
			y[i + u * blockDim.x] = e;
			s += e;
		}
	}
	for (; i < N; i += blockDim.x) {
		float e = expf(LDCS(&x[i]) - m);
		y[i] = e;
		s += e;
	}
	s = block_reduce_sum(s, smem);
	if (tid == 0) psum[blockIdx.x] = s;
	grid.sync();

	s = 0.f;
	for (int j = tid; j < gridDim.x; j += blockDim.x)
		s += psum[j];
	s = block_reduce_sum(s, smem);
	const float inv = 1.0f / s;

	i = blockIdx.x * stride + tid;
	for (; i + stride - blockDim.x < N; i += stride * gridDim.x) {
		float reg[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			reg[u] = y[i + u * blockDim.x];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			y[i + u * blockDim.x] = reg[u] * inv;
	}
	for (; i < N; i += blockDim.x)
		y[i] *= inv;
}

// 1D softmax: y[i] = exp(x[i] - max) / sum(exp(x[j] - max))
__global__ void softmax_kernel(const float *input, float *output, int N) {
	extern __shared__ float shared[];
	const int tid = threadIdx.x;
	const int lane = tid % 32;
	const int warps = blockDim.x >> 5;
	const int warp = tid >> 5;
	const int stride = blockDim.x * UNROLL;

	float *maxvals = shared;
	float *sumvals = &shared[warps];

	const float *x = input;
	float *y = output;

	// ========== max ==========
	float maxval = -INFINITY;
	int i = tid;

	// thread coarsening
	for (; i + stride - blockDim.x < N; i += stride) {
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			maxval = fmaxf(maxval, x[i + u * blockDim.x]);
	}

	// Handle remaining elements
	for (; i < N; i += blockDim.x)
		maxval = fmaxf(maxval, x[i]);

	// Use warp-level reduction
	maxval = warp_reduce_max(maxval);

	// Store the max value of each warp in smem
	if (lane == 0) maxvals[warp] = maxval;
	__syncthreads();

	// Use the first warp to reduce the max values of all warps
	if (tid == 0) {
		float val = maxvals[0];
		for (int w = 1; w < warps; ++w)
			val = fmaxf(val, maxvals[w]);
		maxvals[0] = val;
	}
	__syncthreads();

	// ========== sum（exp + reduce）==========
	float offset = maxvals[0];
	float sumval = 0.0f;
	i = tid;

	// thread coarsening
	for (; i + stride - blockDim.x < N; i += stride) {
		float reg[UNROLL];
		// load x into registers to avoid multiple loads from global memory
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			reg[u] = LDCS(&x[i + u * blockDim.x]);

		// compute exp and store to y, accumulate sum
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			float out = expf(reg[u] - offset);
			y[i + u * blockDim.x] = out;
			sumval += out;
		}
	}

	// Handle remaining elements
	for (; i < N; i += blockDim.x) {
		float out = expf(LDCS(&x[i]) - offset);
		y[i] = out;
		sumval += out;
	}

	// Use warp-level reduction to sum the values
	sumval = warp_reduce_sum(sumval);

	// Store the sum value of each warp in smem
	if (lane == 0) sumvals[warp] = sumval;
	__syncthreads();

	// Use the first warp to reduce the sum values of all warps
	if (tid == 0) {
		float val = sumvals[0];
		for (int w = 1; w < warps; ++w)
			val += sumvals[w];
		sumvals[0] = val;
	}
	__syncthreads();

	// ========== normalize ==========
	float inv_sum = 1.0f / sumvals[0];
	i = tid;

	// thread coarsening
	for (; i + stride - blockDim.x < N; i += stride) {
		// load y into registers to avoid multiple loads from global memory
		float reg[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			reg[u] = y[i + u * blockDim.x];

		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			y[i + u * blockDim.x] = reg[u] * inv_sum;
	}

	// Handle remaining elements
	for (; i < N; i += blockDim.x)
		y[i] = y[i] * inv_sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
	if (N <= 0)
		return;

#if defined(MULTI_FUSED)
	const int threads = 256;
	size_t smem = (threads / 32) * sizeof(float);

	static int max_blocks = -1;
	if (max_blocks < 0) {
		int nb_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&nb_per_sm, softmax_multi_fused, threads, smem);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_blocks = nb_per_sm * prop.multiProcessorCount;
		if (max_blocks < 1) max_blocks = 1;
	}

	int blocks = (N + threads - 1) / threads;
	if (blocks > 1024) blocks = 1024;
	if (blocks > max_blocks) blocks = max_blocks;
	if (blocks < 1) blocks = 1;

	float *pmax, *psum;
	cudaMalloc(&pmax, blocks * sizeof(float));
	cudaMalloc(&psum, blocks * sizeof(float));

	void* args[] = {
		(void*)&input, (void*)&output, (void*)&pmax, (void*)&psum, (void*)&N
	};
	cudaLaunchCooperativeKernel((void*)softmax_multi_fused,
								dim3(blocks), dim3(threads), args, smem, 0);

	cudaFree(pmax);
	cudaFree(psum);
	cudaDeviceSynchronize();
#else
	// Determine the number of threads per block
	int threadsPerBlock = 256;
	if (N < threadsPerBlock) {
		threadsPerBlock = ((N + 31) / 32) * 32;
		if (threadsPerBlock < 32)
			threadsPerBlock = 32;
	}

	int warps = threadsPerBlock / 32;
	size_t smem = 2 * warps * sizeof(float);

	softmax_kernel<<<1, threadsPerBlock, smem>>>(input, output, N);
	cudaDeviceSynchronize();
#endif
}

// solve is the entry point for the GPU implementation of softmax
extern "C" void solve(const float *input, float *output, int N);

// check CUDA errors and exit if any
static void check(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

// CPU implementation of softmax for verification
static void softmax_cpu(const float *x, float *y, int N) {
    float m = x[0];
    for (int i = 1; i < N; ++i) m = fmaxf(m, x[i]);
    float s = 0.f;
    for (int i = 0; i < N; ++i) {
        y[i] = expf(x[i] - m);
        s += y[i];
    }
    for (int i = 0; i < N; ++i) y[i] /= s;
}

// Host-side wall-clock timer (ms)
struct Timer {
	using clock = std::chrono::high_resolution_clock;
	clock::time_point t0, t1;

	void start() { t0 = clock::now(); }
	void stop()  { t1 = clock::now(); }
	float elapsed() const {
		return std::chrono::duration<float, std::milli>(t1 - t0).count();
	}
};


// main function to test the softmax implementation
int main(int argc, char **argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;

    std::vector<float> h_in(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = sinf(i * 0.01f) * 3.f;

    float *d_in = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in, N * sizeof(float)), "malloc in");
    check(cudaMalloc(&d_out, N * sizeof(float)), "malloc out");
    check(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

	const int warmup = 10;
	const int reps = 100;

	// GPU warmup: discard first launches (JIT / driver / cache)
	for (int i = 0; i < warmup; ++i)
		solve(d_in, d_out, N);
	check(cudaDeviceSynchronize(), "warmup sync");

	cudaEvent_t start, stop;
	check(cudaEventCreate(&start), "create start event");
	check(cudaEventCreate(&stop), "create stop event");
	check(cudaEventRecord(start), "record start event");
	for (int i = 0; i < reps; ++i)
		solve(d_in, d_out, N);
	float milliseconds = 0;
	check(cudaEventRecord(stop), "record stop event");
	check(cudaEventSynchronize(stop), "synchronize stop event");
	check(cudaEventElapsedTime(&milliseconds, start, stop), "calculate elapsed time");
	printf("GPU softmax time: %.4f ms (avg of %d)\n", milliseconds / reps, reps);

	check(cudaGetLastError(), "kernel");
	check(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

	// CPU warmup
	for (int i = 0; i < warmup; ++i)
		softmax_cpu(h_in.data(), h_ref.data(), N);

	Timer timer;
	timer.start();
	for (int i = 0; i < reps; ++i)
		softmax_cpu(h_in.data(), h_ref.data(), N);
	timer.stop();
	printf("CPU softmax time: %.4f ms (avg of %d)\n", timer.elapsed() / reps, reps);

    float max_abs = 0.f, sum = 0.f;
    for (int i = 0; i < N; ++i) {
        max_abs = fmaxf(max_abs, fabsf(h_out[i] - h_ref[i]));
        sum += h_out[i];
    }
    printf("N=%d  max|gpu-cpu|=%.6e  sum(gpu)=%.6f (expect ~1)\n", N, max_abs, sum);

    cudaFree(d_in);
    cudaFree(d_out);
    return max_abs > 1e-4f ? 1 : 0;
}