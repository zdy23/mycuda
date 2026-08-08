#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

/*
Q:  M × d
K:  N × d
V:  N × d
O:  M × d

Pipeline:
  1) K^T
  2) score = Q @ K^T / sqrt(d)
  3) row softmax(score)
  4) O = score @ V
*/

#define LOG2_E 1.4426950408889634f

template <int TS, int TPT>
__global__ void matmul_kernel(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int R, int K, float inv_scale
) {
	__shared__ float As[TS][TS];
	__shared__ float Bs[TS][TS];

	int row     = TS * blockIdx.y + threadIdx.y;
	int colBase = TPT * TS * blockIdx.x + threadIdx.x;

	float sum[TPT];
	#pragma unroll
	for (int i = 0; i < TPT; i++) sum[i] = 0.0f;

	int nPhases = (R + TS - 1) / TS;
	for (int phase = 0; phase < nPhases; phase++) {
		int aCol = phase * TS + threadIdx.x;
		As[threadIdx.y][threadIdx.x] =
			(row < M && aCol < R) ? A[row * R + aCol] : 0.0f;

		#pragma unroll
		for (int i = 0; i < TPT; i++) {
			int col  = colBase + i * TS;
			int bRow = phase * TS + threadIdx.y;
			Bs[threadIdx.y][threadIdx.x] =
				(bRow < R && col < K) ? B[bRow * K + col] : 0.0f;
			__syncthreads();

			#pragma unroll
			for (int j = 0; j < TS; j++)
				sum[i] += As[threadIdx.y][j] * Bs[j][threadIdx.x];
			__syncthreads();
		}
	}

	#pragma unroll
	for (int i = 0; i < TPT; i++) {
		int col = colBase + i * TS;
		if (row < M && col < K)
			C[row * K + col] = sum[i] * inv_scale;
	}
}

template <int TPB>
__global__ void row_softmax_kernel(float* __restrict__ input, int M, int N) {
	constexpr int NUM_WARPS = TPB / 32;
	static_assert(NUM_WARPS * 32 == TPB, "TPB multiple of 32");

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

template <int TS>
__global__ void transpose_kernel(
	const float* __restrict__ input, float* __restrict__ output,
	int rows, int cols
) {
	__shared__ float tile[TS][TS + 1];

	int x = blockIdx.x * TS + threadIdx.x;
	int y = blockIdx.y * TS + threadIdx.y;
	if (y < rows && x < cols)
		tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
	__syncthreads();

	x = blockIdx.y * TS + threadIdx.x;
	y = blockIdx.x * TS + threadIdx.y;
	if (y < cols && x < rows)
		output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

extern "C" void solve(const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	constexpr int TS  = 16;
	constexpr int TPT = 2;
	constexpr int TPB = 256;

	static float* d_score = nullptr;
	static float* d_kt    = nullptr;
	static size_t cap_score = 0;
	static size_t cap_kt    = 0;

	size_t need_score = (size_t)M * (size_t)N * sizeof(float);
	size_t need_kt    = (size_t)d * (size_t)N * sizeof(float);
	if (need_score > cap_score) {
		if (d_score) cudaFree(d_score);
		cudaMalloc(&d_score, need_score);
		cap_score = need_score;
	}
	if (need_kt > cap_kt) {
		if (d_kt) cudaFree(d_kt);
		cudaMalloc(&d_kt, need_kt);
		cap_kt = need_kt;
	}

	dim3 block(TS, TS);
	float inv_scale = 1.0f / sqrtf((float)d);

	{
		dim3 grid((d + TS - 1) / TS, (N + TS - 1) / TS);
		transpose_kernel<TS><<<grid, block>>>(K, d_kt, N, d);
	}
	{
		dim3 grid((N + TS * TPT - 1) / (TS * TPT),
				  (M + TS - 1) / TS);
		matmul_kernel<TS, TPT><<<grid, block>>>(
			Q, d_kt, d_score, M, d, N, inv_scale);
	}

	row_softmax_kernel<TPB><<<M, TPB>>>(d_score, M, N);

	{
		dim3 grid((d + TS * TPT - 1) / (TS * TPT),
				  (M + TS - 1) / TS);
		matmul_kernel<TS, TPT><<<grid, block>>>(
			d_score, V, output, M, N, d, 1.0f);
	}
}
