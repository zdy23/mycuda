#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

/*
Q:  M × d      (M 个 query，每个 d 维)
K:  N × d      (N 个 key，每个 d 维)
V:  N × d      (N 个 value，每个 d 维)
O:  M × d      (M 个输出，每个 d 维)

模板参数：
    TS  = TILE_SIZE         矩阵乘法的 tile 边长
    TPT = TILES_PER_THREAD  线程粗化因子（一个线程算几个输出列 tile）
    TPB = THREADS_PER_BLOCK softmax 核的每 block 线程数
*/

// A * B = C
template <int TS, int TPT>
__global__ void matrix_multi_kernel(
	const float* A, const float* B, float* C,
	int M, int N, int K, const float scale
) {
	__shared__ float As[TS][TS];
	__shared__ float Bs[TS][TS];

	int row = TS * blockIdx.y + threadIdx.y;
	int col = TPT * TS * blockIdx.x + threadIdx.x;

	float sum[TPT];
	#pragma unroll
	for (int i = 0; i < TPT; i++) {
		sum[i] = 0.0f;
	}

	for (int phase = 0; phase < (N + TS - 1) / TS; phase++) {
		As[threadIdx.y][threadIdx.x] =
			(row < M && (phase * TS + threadIdx.x) < N) ?
			A[row * N + phase * TS + threadIdx.x] : 0.0f;

		#pragma unroll
		for (int i = 0; i < TPT; i++) {
			int col_i = col + i * TS;
			Bs[threadIdx.y][threadIdx.x] =
				((phase * TS + threadIdx.y) < N && col_i < K) ?
				B[(phase * TS + threadIdx.y) * K + col_i] : 0.0f;
			__syncthreads();

			#pragma unroll
			for (int j = 0; j < TS; j++) {
				sum[i] += As[threadIdx.y][j] * Bs[j][threadIdx.x];
			}
			__syncthreads();
		}
	}

	#pragma unroll
	for (int i = 0; i < TPT; i++) {
		int col_i = col + i * TS;
		if (row < M && col_i < K) {
			C[row * K + col_i] = sum[i] / scale;
		}
	}
}

template <int TPB>
__global__ void row_softmax_kernel(
	float* input, int M, int N
) {
	int row = blockIdx.x;
	int col = threadIdx.x;

	if (row < M) {
		__shared__ float s_max[TPB];
		__shared__ float s_sum[TPB];

		float max_val = -FLT_MAX;
		float sum_val = 0.0f;

		#pragma unroll
		for (int i = col; i < N; i += TPB) {
			float val = input[row * N + i];
			if (val > max_val) {
				sum_val = sum_val * expf(max_val - val) + 1.0f;
				max_val = val;
			} else {
				sum_val += expf(val - max_val);
			}
		}

		s_max[col] = max_val;
		s_sum[col] = sum_val;
		__syncthreads();

		#pragma unroll
		for (int stride = TPB / 2; stride > 0; stride >>= 1) {
			if (col < stride) {
				float thread_max = s_max[col];
				float thread_sum = s_sum[col];
				float peer_max = s_max[col + stride];
				float peer_sum = s_sum[col + stride];

				if (peer_max > -FLT_MAX) {
					if (peer_max > thread_max) {
						s_sum[col] = thread_sum * expf(thread_max - peer_max) + peer_sum;
						s_max[col] = peer_max;
					} else {
						s_sum[col] = peer_sum * expf(peer_max - thread_max) + thread_sum;
					}
				}
			}
			__syncthreads();
		}

		float row_max = s_max[0];
		float row_sum = s_sum[0];

		for (int i = col; i < N; i += TPB) {
			float val = input[row * N + i];
			input[row * N + i] = expf(val - row_max) / row_sum;
		}
	}
}

template <int TS>
__global__ void transpose_kernel(
	const float* input, float* output, int rows, int cols
) {
	int col = blockIdx.x * TS + threadIdx.x;
	int row = blockIdx.y * TS + threadIdx.y;
	if (row < rows && col < cols) {
		output[col * rows + row] = input[row * cols + col];
	}
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d
) {
	const int TS = 16;
	const int TPT = 4;
	const int TPB = 512;
	float* d_score;
	float* d_kt;
	cudaMalloc(&d_score, M * N * sizeof(float));
	cudaMalloc(&d_kt, d * N * sizeof(float));

	dim3 block(TS, TS);

	dim3 grid_trans((d + TS - 1) / TS, (N + TS - 1) / TS);
	transpose_kernel<TS><<<grid_trans, block>>>(K, d_kt, N, d);

	float scale = sqrtf((float)d);

	dim3 grid_score((N + TS * TPT - 1) / (TS * TPT),
					(M + TS - 1) / TS);
	matrix_multi_kernel<TS, TPT><<<grid_score, block>>>(
		Q, d_kt, d_score, M, d, N, scale);

	row_softmax_kernel<TPB><<<M, TPB>>>(d_score, M, N);

	dim3 grid_output((d + TS * TPT - 1) / (TS * TPT),
					 (M + TS - 1) / TS);
	matrix_multi_kernel<TS, TPT><<<grid_output, block>>>(
		d_score, V, output, M, N, d, 1.0f);

	cudaFree(d_score);
	cudaFree(d_kt);
}
