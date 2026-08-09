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
		__syncthreads();

		// load Bs
		#pragma unroll

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