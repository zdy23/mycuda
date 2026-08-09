#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define LOG2_E 1.4426950408889634f

// Matrix mul kernel with shared memory tiling and register blocking
template <int TS, int TM, int TN, bool TRANSPOSED_B = false>
__global__ void matmul_kernel(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int R, int K, float inv_scale
) {
	__shared__ float As[TS][TS + 1];
	__shared__ float Bs[TS][TS];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int rowBlock = TS * by;
	int colBlock = TS * bx;
	int row0     = rowBlock + ty * TM;
	int col0     = colBlock + tx * TN;

	// init acc
	float acc[TM][TN];
	#pragma unroll
	for (int r = 0; r < TM; ++r)
		#pragma unroll
		for (int c = 0; c < TN; ++c)
			acc[r][c] = 0.0f;

	// tid and threadsPerBlock
	int tid  = threadIdx.y * blockDim.x + threadIdx.x;
	int nTh  = blockDim.x * blockDim.y;

	// 
	int nPhases = (R + TS - 1) / TS;
	for (int phase = 0; phase < nPhases; ++phase) {
		int kBase = phase * TS;

		for (int i = tid; i < TS * TS; i += nTh) {
			int r = i / TS;
			int c = i % TS;
			int gRow = rowBlock + r;
			int gKC  = kBase + c;
			As[r][c] = (gRow < M && gKC < R) ? A[gRow * R + gKC] : 0.0f;
		}

		for (int i = tid; i < TS * TS; i += nTh) {
			int k = i / TS;
			int c = i % TS;
			int gKR  = kBase + k;
			int gCol = colBlock + c;
			Bs[k][c] = (gKR < R && gCol < K)
				? (TRANSPOSED_B ? B[gCol * R + gKR] : B[gKR * K + gCol])
				: 0.0f;
		}
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < TS; ++k) {
			float b[TN];
			#pragma unroll
			for (int c = 0; c < TN; ++c)
				b[c] = Bs[k][tx * TN + c];

			#pragma unroll
			for (int r = 0; r < TM; ++r) {
				float a = As[ty * TM + r][k];
				#pragma unroll
				for (int c = 0; c < TN; ++c)
					acc[r][c] += a * b[c];
			}
		}
		__syncthreads();
	}

	#pragma unroll
	for (int r = 0; r < TM; ++r) {
		int gRow = row0 + r;
		if (gRow >= M) continue;
		#pragma unroll
		for (int c = 0; c < TN; ++c) {
			int gCol = col0 + c;
			if (gCol < K)
				C[gRow * K + gCol] = acc[r][c] * inv_scale;
		}
	}
}

// Row-wise softmax kernel
template <int TPB>
__global__ void row_softmax_kernel(float* __restrict__ input, int M, int N) {
	constexpr int NUM_WARPS = TPB / 32;
	int row  = blockIdx.x;
	if (row >= M) return;

	int tid  = threadIdx.x;
	int warp = tid >> 5;
	int lane = tid & 31;

	__shared__ float s_warp_max[NUM_WARPS];
	__shared__ float s_warp_sum[NUM_WARPS];

	float max_val = -FLT_MAX;
	float sum_val = 0.0f;

	for (int i = tid; i < N; i += TPB) {
		float v = input[row * N + i];
		if (v > max_val) {
			sum_val = sum_val * exp2f((max_val - v) * LOG2_E) + 1.0f;
			max_val = v;
		} else {
			sum_val += exp2f((v - max_val) * LOG2_E);
		}
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		float pm = __shfl_xor_sync(0xffffffff, max_val, off);
		float ps = __shfl_xor_sync(0xffffffff, sum_val, off);
		if (pm > max_val) {
			sum_val = sum_val * exp2f((max_val - pm) * LOG2_E) + ps;
			max_val = pm;
		} else {
			sum_val = ps * exp2f((pm - max_val) * LOG2_E) + sum_val;
		}
	}

	if (lane == 0) {
		s_warp_max[warp] = max_val;
		s_warp_sum[warp] = sum_val;
	}
	__syncthreads();

	if (warp == 0) {
		float wm = (lane < NUM_WARPS) ? s_warp_max[lane] : -FLT_MAX;
		float ws = (lane < NUM_WARPS) ? s_warp_sum[lane] : 0.0f;

		#pragma unroll
		for (int off = NUM_WARPS / 2; off > 0; off >>= 1) {
			float pm = __shfl_xor_sync(0xffffffff, wm, off);
			float ps = __shfl_xor_sync(0xffffffff, ws, off);
			if (pm > wm) {
				ws = ws * exp2f((wm - pm) * LOG2_E) + ps;
				wm = pm;
			} else {
				ws = ps * exp2f((pm - wm) * LOG2_E) + ws;
			}
		}
		if (lane == 0) {
			s_warp_max[0] = wm;
			s_warp_sum[0] = ws;
		}
	}
	__syncthreads();

	float row_max = s_warp_max[0];
	float inv_sum = 1.0f / s_warp_sum[0];

	for (int i = tid; i < N; i += TPB) {
		float v = input[row * N + i];
		input[row * N + i] = exp2f((v - row_max) * LOG2_E) * inv_sum;
	}
}

extern "C" void solve(const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	constexpr int TS  = 32;
	constexpr int TM  = 2;
	constexpr int TN  = 2;
	constexpr int TPB = 256;

	static float* d_score = nullptr;
	static size_t cap_score = 0;

	size_t need_score = (size_t)M * (size_t)N * sizeof(float);
	if (need_score > cap_score) {
		if (d_score) cudaFree(d_score);
		cudaMalloc(&d_score, need_score);
		cap_score = need_score;
	}

	dim3 block(TS / TN, TS / TM);
	float inv_scale = 1.0f / sqrtf((float)d);

	{
		dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
		matmul_kernel<TS, TM, TN, true><<<grid, block>>>(
			Q, K, d_score, M, d, N, inv_scale);
	}

	row_softmax_kernel<TPB><<<M, TPB>>>(d_score, M, N);

	{
		dim3 grid((d + TS - 1) / TS, (M + TS - 1) / TS);
		matmul_kernel<TS, TM, TN, false><<<grid, block>>>(
			d_score, V, output, M, N, d, 1.0f);
	}
}