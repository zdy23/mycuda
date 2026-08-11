#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define LOG2_E 1.4426950408889634f

__device__ __forceinline__ float warp_reduce_sum(float v) {
	#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		v += __shfl_xor_sync(0xffffffff, v, o);
	return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
	#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
	return v;
}


template <int TPB>
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	constexpr int NW = TPB / 32;
	v = warp_reduce_sum(v);
	if (lane == 0)
		smem[warp] = v;
	__syncthreads();
	v = (lane < NW) ? smem[lane] : 0.f;
	if (warp == 0)
		v = warp_reduce_sum(v);
	return v;
}

template <int TPB>
__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	constexpr int NW = TPB / 32;
	v = warp_reduce_max(v);
	if (lane == 0)
		smem[warp] = v;
	__syncthreads();
	v = (lane < NW) ? smem[lane] : -FLT_MAX;
	if (warp == 0)
		v = warp_reduce_max(v);
	return v;
}

template <int BR, int BC, int TPB>
__global__ void flash_attn_kernel(
	const float* __restrict__ Q,
	const float* __restrict__ K,
	const float* __restrict__ V,
	float* __restrict__ Out,
	int M, int N, int d, float scale
) {
	extern __shared__ float smem[];

	float* sQ = smem;
	float* sK = sQ + BR * d;
	float* sV = sK + BC * d;
	float* sS = sV + BC * d;
	float* sO = sS + BR * BC;
	float* sM = sO + BR * d;
	float* sL = sM + BR;
	float* sRed = sL + BR;

	const int tid = threadIdx.x;
	const int q0 = blockIdx.x * BR;

	for (int i = tid; i < BR; i += TPB) {
		sM[i] = -FLT_MAX;
		sL[i] = 0.f;
	}
	for (int i = tid; i < BR * d; i += TPB) {
		const int r = i / d;
		const int c = i % d;
		const int gr = q0 + r;
		sO[i] = 0.f;
		sQ[i] = (gr < M) ? Q[gr * d + c] : 0.f;
	}
	__syncthreads();

	for (int j0 = 0; j0 < N; j0 += BC) {
		for (int i = tid; i < BC * d; i += TPB) {
			const int r = i / d;
			const int c = i % d;
			const int gr = j0 + r;
			const bool valid = (gr < N);
			sK[i] = valid ? K[gr * d + c] : 0.f;
			sV[i] = valid ? V[gr * d + c] : 0.f;
		}
		__syncthreads();

		for (int i = tid; i < BR * BC; i += TPB) {
			const int r = i / BC;
			const int c = i % BC;
			const int gr = q0 + r;
			const int gc = j0 + c;
			if (gr < M && gc < N) {
				float acc = 0.f;
				const float* qrow = sQ + r * d;
				const float* krow = sK + c * d;
				int kk = 0;
				for (; kk + 3 < d; kk += 4) {
					acc += qrow[kk] * krow[kk];
					acc += qrow[kk + 1] * krow[kk + 1];
					acc += qrow[kk + 2] * krow[kk + 2];
					acc += qrow[kk + 3] * krow[kk + 3];
				}
				for (; kk < d; ++kk)
					acc += qrow[kk] * krow[kk];
				sS[i] = acc * scale;
			} else {
				sS[i] = -FLT_MAX;
			}
		}
		__syncthreads();

		for (int r = 0; r < BR; ++r) {
			const int gr = q0 + r;
			if (gr >= M) {
				__syncthreads();
				continue;
			}

			float tmax = -FLT_MAX;
			for (int c = tid; c < BC; c += TPB) {
				if (j0 + c < N)
					tmax = fmaxf(tmax, sS[r * BC + c]);
			}
			tmax = block_reduce_max<TPB>(tmax, sRed);
			if (tid == 0)
				sRed[8] = tmax;
			__syncthreads();
			const float m_prev = sM[r];
			const float m_new = fmaxf(m_prev, sRed[8]);
			const float alpha = (m_prev == -FLT_MAX) ? 0.f : exp2f((m_prev - m_new) * LOG2_E);

			float tsum = 0.f;
			for (int c = tid; c < BC; c += TPB) {
				if (j0 + c < N) {
					const float p = exp2f((sS[r * BC + c] - m_new) * LOG2_E);
					sS[r * BC + c] = p;
					tsum += p;
				} else {
					sS[r * BC + c] = 0.f;
				}
			}
			tsum = block_reduce_sum<TPB>(tsum, sRed);
			if (tid == 0) {
				sRed[8] = tsum;
				sRed[9] = alpha;
				sM[r] = m_new;
				sL[r] = sL[r] * alpha + tsum;
			}
			__syncthreads();
			const float a = sRed[9];

			float* orow = sO + r * d;
			for (int kk = tid; kk < d; kk += TPB) {
				float acc = orow[kk] * a;
				#pragma unroll 4
				for (int c = 0; c < BC; ++c)
					acc += sS[r * BC + c] * sV[c * d + kk];
				orow[kk] = acc;
			}
			__syncthreads();
		}
	}

	for (int i = tid; i < BR * d; i += TPB) {
		const int r = i / d;
		const int c = i % d;
		const int gr = q0 + r;
		if (gr < M) {
			const float inv = 1.f / sL[r];
			Out[gr * d + c] = sO[i] * inv;
		}
	}
}

template <int TS, int TM, int TN, bool TRANSPOSED_B>
__global__ void matmul_vec_kernel(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int R, int K, float inv_scale
) {
	__shared__ float As[TS][TS + 1];
	__shared__ float Bs[TS][TS + 1];

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	const int rowBlock = TS * by;
	const int colBlock = TS * bx;
	const int row0 = rowBlock + ty * TM;
	const int col0 = colBlock + tx * TN;

	float acc[TM][TN];
	#pragma unroll
	for (int r = 0; r < TM; ++r)
		#pragma unroll
		for (int c = 0; c < TN; ++c)
			acc[r][c] = 0.f;

	const int tid = ty * blockDim.x + tx;
	const int nTh = blockDim.x * blockDim.y;
	const int nPhases = (R + TS - 1) / TS;

	for (int phase = 0; phase < nPhases; ++phase) {
		const int kBase = phase * TS;

		for (int i = tid; i < TS * TS; i += nTh) {
			const int r = i / TS;
			const int c = i % TS;
			const int gRow = rowBlock + r;
			const int gKC = kBase + c;
			As[r][c] = (gRow < M && gKC < R) ? A[gRow * R + gKC] : 0.f;
		}

		for (int i = tid; i < TS * TS; i += nTh) {
			const int k = i / TS;
			const int c = i % TS;
			const int gKR = kBase + k;
			const int gCol = colBlock + c;
			float v = 0.f;
			if (gKR < R && gCol < K) {
				if constexpr (TRANSPOSED_B)
					v = B[gCol * R + gKR];
				else
					v = B[gKR * K + gCol];
			}
			Bs[k][c] = v;
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
				const float a = As[ty * TM + r][k];
				#pragma unroll
				for (int c = 0; c < TN; ++c)
					acc[r][c] = fmaf(a, b[c], acc[r][c]);
			}
		}
		__syncthreads();
	}

	#pragma unroll
	for (int r = 0; r < TM; ++r) {
		const int gRow = row0 + r;
		if (gRow >= M)
			continue;
		#pragma unroll
		for (int c = 0; c < TN; ++c) {
			const int gCol = col0 + c;
			if (gCol < K)
				C[gRow * K + gCol] = acc[r][c] * inv_scale;
		}
	}
}

template <int TPB>
__global__ void row_softmax_kernel(float* __restrict__ input, int M, int N) {
	constexpr int NUM_WARPS = TPB / 32;
	const int row = blockIdx.x;
	if (row >= M)
		return;

	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;

	__shared__ float s_warp_max[NUM_WARPS];
	__shared__ float s_warp_sum[NUM_WARPS];

	float max_val = -FLT_MAX;
	float sum_val = 0.f;

	for (int i = tid; i < N; i += TPB) {
		const float v = input[row * N + i];
		if (v > max_val) {
			sum_val = sum_val * exp2f((max_val - v) * LOG2_E) + 1.f;
			max_val = v;
		} else {
			sum_val += exp2f((v - max_val) * LOG2_E);
		}
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		const float pm = __shfl_xor_sync(0xffffffff, max_val, off);
		const float ps = __shfl_xor_sync(0xffffffff, sum_val, off);
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
		float ws = (lane < NUM_WARPS) ? s_warp_sum[lane] : 0.f;
		#pragma unroll
		for (int off = NUM_WARPS / 2; off > 0; off >>= 1) {
			const float pm = __shfl_xor_sync(0xffffffff, wm, off);
			const float ps = __shfl_xor_sync(0xffffffff, ws, off);
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

	const float row_max = s_warp_max[0];
	const float inv_sum = 1.f / s_warp_sum[0];

	for (int i = tid; i < N; i += TPB) {
		const float v = input[row * N + i];
		input[row * N + i] = exp2f((v - row_max) * LOG2_E) * inv_sum;
	}
}

static inline size_t flash_smem_bytes(int BR, int BC, int d) {
	const size_t nfloat =
		(size_t)BR * d +
		(size_t)BC * d +
		(size_t)BC * d +
		(size_t)BR * BC +
		(size_t)BR * d +
		(size_t)BR +
		(size_t)BR +
		16;
	return nfloat * sizeof(float);
}

template <int BR, int BC, int TPB>
static void launch_flash(
	const float* Q, const float* K, const float* V, float* output,
	int M, int N, int d, float scale
) {
	const size_t smem = flash_smem_bytes(BR, BC, d);
	auto* fn = flash_attn_kernel<BR, BC, TPB>;
	cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
	const int grid = (M + BR - 1) / BR;
	fn<<<grid, TPB, smem>>>(Q, K, V, output, M, N, d, scale);
}

extern "C" void solve(
	const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	const float scale = 1.f / sqrtf((float)d);
	constexpr int TPB = 256;
	const size_t lim = 96 * 1024;

	if (flash_smem_bytes(32, 32, d) <= lim && M > 0 && N > 0 && d > 0) {
		launch_flash<32, 32, TPB>(Q, K, V, output, M, N, d, scale);
		return;
	}
	if (flash_smem_bytes(16, 32, d) <= lim && M > 0 && N > 0 && d > 0) {
		launch_flash<16, 32, TPB>(Q, K, V, output, M, N, d, scale);
		return;
	}
	if (flash_smem_bytes(8, 32, d) <= lim && M > 0 && N > 0 && d > 0) {
		launch_flash<8, 32, TPB>(Q, K, V, output, M, N, d, scale);
		return;
	}

	constexpr int TS = 32;
	constexpr int TM = 4;
	constexpr int TN = 4;
	constexpr int SOFT_TPB = 256;

	static float* d_score = nullptr;
	static size_t cap_score = 0;

	const size_t need_score = (size_t)M * (size_t)N * sizeof(float);
	if (need_score > cap_score) {
		if (d_score)
			cudaFree(d_score);
		cudaMalloc(&d_score, need_score);
		cap_score = need_score;
	}

	dim3 block(TS / TN, TS / TM);

	{
		dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
		matmul_vec_kernel<TS, TM, TN, true><<<grid, block>>>(
			Q, K, d_score, M, d, N, scale);
	}

	row_softmax_kernel<SOFT_TPB><<<M, SOFT_TPB>>>(d_score, M, N);

	{
		dim3 grid((d + TS - 1) / TS, (M + TS - 1) / TS);
		matmul_vec_kernel<TS, TM, TN, false><<<grid, block>>>(
			d_score, V, output, M, N, d, 1.f);
	}
}
