#include <__clang_cuda_builtin_vars.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TS 32
#define TM 4
#define TN 4

__global__ void gemm_tiled(
	const half* A, const half* B, half* C,
	int M, int N, int K,
	float alpha, float beta
) {
	__shared__ half As[TS][TS];
	__shared__ half Bs[TS][TS];

	int rowT = threadIdx.x;
	int colT = threadIdx.y;
	int mblk = blockIdx.x;
	int nblk = blockIdx.y;

	int row_base = rowT * TM + mblk * TS;
	int col_base = colT * TN + nblk * TS;

	float acc[TM][TN];
	for (int i = 0; i < TM; ++i) {
		for (int j = 0; j < TN; j++) {
			acc[i][j] = 0.f;
		}
	}

	for (int i = 0; i < (K + TS - 1) / TS; ++i) {
		// load As
		#pragma unroll
		for (int r = 0; r < TM; ++r) {
			#pragma unroll
			for (int c = 0; c < TN; ++c) {
				int a_row  = row_base + r;
				int a_kcol = i * TS + colT * TN + c;
				As[rowT * TM + r][colT * TN + c] =
					(a_row < M && a_kcol < K) ? A[a_row * K + a_kcol]
					: __float2half(0.f);
			}
		}

		// load Bs
		#pragma unroll
		for (int r = 0; r < TM; ++r) {
			#pragma unroll
			for (int c = 0; c < TN; ++c) {
				int b_krow = i * TS + rowT * TM + r;
				int b_col = col_base + c;
				Bs[rowT * TM + r][colT * TN + c] =
					(b_krow < K && b_col < N) ? B[b_krow * N + b_col]
					: __float2half(0.f);
			}
		}
		__syncthreads();

		for (int k = 0; k < TS; ++k) {
			float b[TN];
			#pragma unroll
			for (int c = 0; c < TN; ++c) {
				b[c] = __half2float(Bs[k][colT * TN + c]);
			}

			#pragma unroll
			for (int r = 0; r < TM; ++r) {
				float a = __half2float(As[rowT * TM + r][k]);
				#pragma unroll
				for (int c = 0; c < TN; ++c) {
					acc[r][c] += a * b[c];
				}
			}
		}
		__syncthreads();
	}

	#pragma unroll
	for (int r = 0; r < TM; ++r) {
		int row = row_base + r;
		#pragma unroll
		for (int c = 0; c < TN; ++c) {
			int col = col_base + c;
			if (row < M && col < N) {
				float old = (beta != 0.f) ? beta * __half2float(C[row * N + col]) : 0.f;
				C[row * N + col] = __float2half(alpha * acc[r][c] + old);
			}
		}
	}

}

extern "C" void solve(
	const half* A, const half* B, half* C,
	int M, int N, int K,
	float alpha, float beta
) {
	dim3 block(TS / TM, TS / TN); // (8, 8) 
	dim3 grid(
		(M + TS - 1) / TS,
		(N + TS - 1) / TS
	);
	gemm_tiled<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}