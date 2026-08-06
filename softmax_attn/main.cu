#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

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

__global__ void softmax_attn_single(
	const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	extern __shared__ float smem[];
	float* scores = smem;
	float* reduce_buf = smem + N;

	const int tid = threadIdx.x;
	const int i = blockIdx.x;
	const float inv = rsqrtf((float)d);

	const float* qi = Q + (size_t)i * d;	

	// === pass1: scores + max ===
	float m = -INFINITY;
	for (int j = tid; j < N; j += blockDim.x) {
		float acc = 0.f;
		const float* kj = K + (size_t)j * d;
		for (int k = 0; k < d; k++) {
			acc += qi[k] * kj[k];
		}
		acc *= inv;
		scores[j] = acc;
		m = fmaxf(m, acc);
	}
	m = block_reduce_max(m, reduce_buf);

	// === pass2: exp + sum
	float s = 0.f;
	for (int j = tid; j < N; j += blockDim.x) {
		float e = __expf(scores[j] - m);
		scores[j] = e;
		s += e;
	}
	s = block_reduce_sum(s, reduce_buf);

	// === pass 3: norm & get obj ===
	const float inv_sum = 1.f / s;
	float* oi = output + (size_t)i * d;

	for (int k = tid; k < d; k += blockDim.x) {
		float acc = 0.f;
		for (int j = 0; j < N; j++)
			acc += scores[j] * V[(size_t)j * d + k];
		oi[k] = acc * inv_sum;
	}
}

__global__ void softmax_attn_multi(
	const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	extern __shared__ float smem[];
	float* scores = smem;
	float* reduce_buf = smem + N;

	const int tid = threadIdx.x;
	const float inv = rsqrtf((float)d);

	// === grid-stride ===
	for (int i = blockIdx.x; i < M; i += gridDim.x) {
		const float* qi = Q + (size_t)i * d;
		
		// === pass1: scores + max ===
		float m = -INFINITY;
		for (int j = tid; j < N; j += blockDim.x) {
			float acc = 0.f;
			const float* kj = K + (size_t)j * d;
			for (int k = 0; k < d; k++)
				acc += qi[k] * kj[k];
			acc *= inv;
			scores[j] = acc;
			m = fmaxf(m, acc);
		}
		m = block_reduce_max(m, reduce_buf);

		// === pass2: exp + sum ===
		float s = 0.f;
		for (int j = tid; j < N; j += blockDim.x) {
			float e = __expf(scores[j] - m);
			scores[j] = e;
			s += e;
		}
		s = block_reduce_sum(s, reduce_buf);

		// === pass3: norm + O = P @ V ===
		const float inv_sum = 1.f / s;
		float* oi = output + (size_t)i * d;

		for (int k = tid; k < d; k += blockDim.x) {
			float acc = 0.f;
			for (int j = 0; j < N; j++)
				acc += scores[j] * V[(size_t)j * d + k];
			oi[k] = acc * inv_sum;
		}
	}
}

// Q, K, V, output are device pointers
extern "C" void solve(
	const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	if (M <= 0 || N <= 0 || d <= 0) return;

	const int threads = 256;
	const int warps = threads >> 5;
	// scores[N] + reduce_buf[warps]
	size_t smem = (size_t)(N + warps) * sizeof(float);

#if defined(FORCE_SINGLE)
	softmax_attn_single<<<M, threads, smem>>>(Q, K, V, output, M, N, d);
#elif defined(FORCE_MULTI)
	static int max_blocks = -1;
	if (max_blocks < 0) {
		int nb_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&nb_per_sm, softmax_attn_multi, threads, smem);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_blocks = nb_per_sm * prop.multiProcessorCount;
		if (max_blocks < 1) max_blocks = 1;
	}
	int blocks = M < max_blocks ? M : max_blocks;
	softmax_attn_multi<<<blocks, threads, smem>>>(Q, K, V, output, M, N, d);
#else
	const int SMALL_M = 256;
	if (M <= SMALL_M) {
		softmax_attn_single<<<M, threads, smem>>>(Q, K, V, output, M, N, d);
	} else {
		static int max_blocks = -1;
		if (max_blocks < 0) {
			int nb_per_sm = 0;
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&nb_per_sm, softmax_attn_multi, threads, smem);
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			max_blocks = nb_per_sm * prop.multiProcessorCount;
			if (max_blocks < 1) max_blocks = 1;
		}
		int blocks = M < max_blocks ? M : max_blocks;
		softmax_attn_multi<<<blocks, threads, smem>>>(Q, K, V, output, M, N, d);
	}
#endif
	cudaDeviceSynchronize();
}

static void check(cudaError_t e, const char* msg) {
	if (e != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
		std::exit(1);
	}
}

// O[i] = softmax(Q[i]K^T / sqrt(d)) @ V
static void attn_cpu(
	const float* Q, const float* K, const float* V,
	float* O, int M, int N, int d
) {
	const float inv = 1.f / sqrtf((float)d);
	std::vector<float> scores(N);
	for (int i = 0; i < M; ++i) {
		const float* qi = Q + (size_t)i * d;
		float m = -INFINITY;
		for (int j = 0; j < N; ++j) {
			const float* kj = K + (size_t)j * d;
			float acc = 0.f;
			for (int k = 0; k < d; ++k)
				acc += qi[k] * kj[k];
			acc *= inv;
			scores[j] = acc;
			m = fmaxf(m, acc);
		}
		float s = 0.f;
		for (int j = 0; j < N; ++j) {
			scores[j] = expf(scores[j] - m);
			s += scores[j];
		}
		const float inv_s = 1.f / s;
		float* oi = O + (size_t)i * d;
		for (int k = 0; k < d; ++k) {
			float acc = 0.f;
			for (int j = 0; j < N; ++j)
				acc += scores[j] * V[(size_t)j * d + k];
			oi[k] = acc * inv_s;
		}
	}
}

struct Timer {
	using clock = std::chrono::high_resolution_clock;
	clock::time_point t0, t1;
	void start() { t0 = clock::now(); }
	void stop() { t1 = clock::now(); }
	float elapsed() const {
		return std::chrono::duration<float, std::milli>(t1 - t0).count();
	}
};

int main(int argc, char** argv) {
	int M = (argc > 1) ? std::atoi(argv[1]) : 512;
	int N = (argc > 2) ? std::atoi(argv[2]) : 512;
	int d = (argc > 3) ? std::atoi(argv[3]) : 64;

	const size_t qkv = (size_t)M * d;
	const size_t knv = (size_t)N * d;

	std::vector<float> h_Q(qkv), h_K(knv), h_V(knv), h_O(qkv), h_ref(qkv);
	for (size_t i = 0; i < qkv; ++i)
		h_Q[i] = sinf(i * 0.01f) * 0.5f;
	for (size_t i = 0; i < knv; ++i) {
		h_K[i] = cosf(i * 0.013f) * 0.5f;
		h_V[i] = sinf(i * 0.017f) * 0.5f;
	}

	float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
	check(cudaMalloc(&d_Q, qkv * sizeof(float)), "malloc Q");
	check(cudaMalloc(&d_K, knv * sizeof(float)), "malloc K");
	check(cudaMalloc(&d_V, knv * sizeof(float)), "malloc V");
	check(cudaMalloc(&d_O, qkv * sizeof(float)), "malloc O");
	check(cudaMemcpy(d_Q, h_Q.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D Q");
	check(cudaMemcpy(d_K, h_K.data(), knv * sizeof(float), cudaMemcpyHostToDevice), "H2D K");
	check(cudaMemcpy(d_V, h_V.data(), knv * sizeof(float), cudaMemcpyHostToDevice), "H2D V");

	const int warmup = 5;
	const int reps = 20;

	for (int i = 0; i < warmup; ++i)
		solve(d_Q, d_K, d_V, d_O, M, N, d);
	check(cudaDeviceSynchronize(), "warmup sync");

	cudaEvent_t start, stop;
	check(cudaEventCreate(&start), "create start");
	check(cudaEventCreate(&stop), "create stop");
	check(cudaEventRecord(start), "record start");
	for (int i = 0; i < reps; ++i)
		solve(d_Q, d_K, d_V, d_O, M, N, d);
	check(cudaEventRecord(stop), "record stop");
	check(cudaEventSynchronize(stop), "sync stop");
	float ms = 0.f;
	check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
	printf("GPU attn time: %.4f ms (avg of %d)\n", ms / reps, reps);

	check(cudaGetLastError(), "kernel");
	check(cudaMemcpy(h_O.data(), d_O, qkv * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

	// CPU：大 shape 只做 correctness 1 次，小 shape 才 bench
	const long long flops_proxy = (long long)M * N * d;
	const bool bench_cpu = flops_proxy <= 512ll * 512 * 128;

	if (bench_cpu) {
		for (int i = 0; i < 2; ++i)
			attn_cpu(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), M, N, d);
		Timer timer;
		timer.start();
		for (int i = 0; i < 5; ++i)
			attn_cpu(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), M, N, d);
		timer.stop();
		printf("CPU attn time: %.4f ms (avg of %d)\n", timer.elapsed() / 5, 5);
	} else {
		attn_cpu(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), M, N, d);
		printf("CPU attn time: 0.0000 ms (skipped bench)\n");
	}

	float max_abs = 0.f;
	for (size_t i = 0; i < qkv; ++i)
		max_abs = fmaxf(max_abs, fabsf(h_O[i] - h_ref[i]));
	printf("M=%d N=%d d=%d  max|gpu-cpu|=%.6e\n", M, N, d, max_abs);

	cudaFree(d_Q);
	cudaFree(d_K);
	cudaFree(d_V);
	cudaFree(d_O);
	return max_abs > 1e-3f ? 1 : 0;
}