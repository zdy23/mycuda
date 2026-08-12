#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>


__device__ __forceinline__ uint32_t smem_u32(const void *p) {
	return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_f16(
	float *d, const uint32_t *a, const uint32_t *b, const float *c
) {
	asm volatile(
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		"{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
		: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
		  "r"(b[0]), "r"(b[1]),
		  "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}


template <int N>
__device__ __forceinline__ void cp_async_wait() {
	if constexpr (N == 0)
		asm volatile("cp.async.wait_all;\n" ::);
	else
		asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void cp_async16(half *smem, const half *gmem) {
	uint32_t sp = smem_u32(smem);
	asm volatile(
		"cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
		:: "r"(sp), "l"(gmem));
}


__device__ __forceinline__ void ldmatrix_x4(half *smem, uint32_t *dst) {
	uint32_t sp = smem_u32(smem);
	asm volatile(
		"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
		: "r"(sp));
}

__device__ __forceinline__ void ldmatrix_x4_t(half *smem, uint32_t *dst) {
	uint32_t sp = smem_u32(smem);
	asm volatile(
		"ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
		: "r"(sp));
}

__device__ __forceinline__ void load_a_frags(
	half *sA, uint32_t *reg_a, int lane, int warp_row, int k_step
) {
	int col = k_step * 2 + (lane / 16);
	int r0 = (k_step & 1) * 16;

	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		int row = (lane % 16) + warp_row * 16 + i * 32;
		int col_sw = (row % 8) ^ col;
		int base = row * 64 + col_sw * 8;
		ldmatrix_x4(sA + base, reg_a + r0 + i * 4);
	}
}

__device__ __forceinline__ void load_b_frags(
	half *sB, uint32_t *reg_b, int lane, int warp_col, int k_step
) {
	int row = (lane % 16) + k_step * 16;
	int col0 = (lane / 16) * 2 + warp_col;
	int r0 = (k_step & 1) * 16;

	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		int col = col0 + i * 4;
		int col_sw = (row % 8) ^ col;
		int off = row * 128 + col_sw * 8;
		ldmatrix_x4_t(sB + off, reg_b + r0 + i * 4);
	}
}

__device__ __forceinline__ void mma_tile(
	int nM, int nN, uint32_t *reg_a, uint32_t *reg_b, float *reg_c, int stage
) {
	int a0 = (stage & 1) * 16;
	#pragma unroll
	for (int m = 0; m < nM; ++m) {
		#pragma unroll
		for (int n = 0; n < nN; ++n) {
			float *c = reg_c + (m * 8 + n) * 4;
			mma_f16(c, reg_a + a0 + m * 4, reg_b + a0 + n * 2, c);
		}
	}
}

template <int BM, int BN, int BK, int NTHREADS, int NPIPE>
__global__ void gemm_tc_kernel(
	const half *__restrict__ A,
	const half *__restrict__ B,
	half *__restrict__ C,
	int M, int N, int K,
	float alpha, float beta
) {
	int tid = threadIdx.x;
	int warp = tid / 32;
	int lane = tid % 32;
	int warp_row = warp % 2;
	int warp_col = warp / 2;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	const half *gA = A + bx * BM * K;
	const half *gB = B + by * BN;
	half *gC = C + bx * BM * N + by * BN;

	extern __shared__ half smem[];
	half *sA = smem;
	half *sB = smem + BM * BK * NPIPE;

	float reg_c[128];
	uint32_t reg_a[32];
	uint32_t reg_b[32];

	for (int i = 0; i < 128; ++i)
		reg_c[i] = 0.0f;

	constexpr int nM = (BM / 2) / 16;
	constexpr int nN = (BN / 2) / 8;
	constexpr int vecK = BK / 8;
	constexpr int vecN = BN / 8;
	constexpr int nKmma = BK / 16;

	int nTiles = (K + BK - 1) / BK;
	int tile = 0;

	int a_row = tid / vecK;
	int a_col = tid % vecK;
	int a_col_sw = (a_row % 8) ^ a_col;
	int a_sm = a_row * BK + a_col_sw * 8;
	int a_gm = a_row * K + a_col * 8;

	int b_row = tid / vecN;
	int b_col = tid % vecN;
	int b_col_sw = (b_row % 8) ^ b_col;
	int b_sm = b_row * BN + b_col_sw * 8;
	int b_gm = b_row * N + b_col * 8;

	// 加载前 NPIPE - 1 个 tile
	#pragma unroll
	for (int p = 0; p < NPIPE - 1; ++p) {
		half *tileA = sA + p * BM * BK;
		half *tileB = sB + p * BK * BN;
		const half *gAt = gA + p * BK;
		const half *gBt = gB + p * BK * N;

		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			cp_async16(tileA + a_sm + i * 16 * BK, gAt + a_gm + i * 16 * K);
		}
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			cp_async16(tileB + b_sm + i * 8 * BN, gBt + b_gm + i * 8 * N);
		}
		asm volatile("cp.async.commit_group;\n" ::);
		--nTiles;
		if (nTiles > 0) {
			++tile;
		}
	}

	// 由于前 NPIPE - 1 个 tile 已经加载到共享内存中，所以可以直接从共享内存中加载数据
	int pipe_r = 0;
	int pipe_w = NPIPE - 1;
	half *curA = sA + pipe_r * BM * BK;
	half *curB = sB + pipe_r * BN * BK;

	// 等待前 NPIPE - 1 个 tile 加载完成
	cp_async_wait<NPIPE - 2>();
	__syncthreads();

	// 加载当前 tile 的数据到寄存器中
	load_a_frags(curA, reg_a, lane, warp_row, 0);
	load_b_frags(curB, reg_b, lane, warp_col, 0);

	// while 用来收尾，当该计算的计算了之后停止
	while (nTiles > -(NPIPE - 1)) {
		#pragma unroll
		for (int kk = 0; kk < nKmma; ++kk) {
			if (kk == nKmma - 1) {
				curA = sA + pipe_r * BM * BK;
				curB = sB + pipe_r * BN * BK;
				cp_async_wait<NPIPE - 2>();
				__syncthreads();
			}

			int kk_next = (kk + 1) % nKmma;
			load_a_frags(curA, reg_a, lane, warp_row, kk_next);
			load_b_frags(curB, reg_b, lane, warp_col, kk_next);
			mma_tile(nM, nN, reg_a, reg_b, reg_c, kk);

			if (kk == 0) {
				half *tileA = sA + pipe_w * BM * BK;
				half *tileB = sB + pipe_w * BN * BK;
				const half *gAt = gA + tile * BK;
				const half *gBt = gB + tile * BK * N;
				#pragma unroll
				for (int i = 0; i < 8; ++i)
					cp_async16(tileA + a_sm + i * 16 * BK, gAt + a_gm + i * 16 * K);
				#pragma unroll
				for (int i = 0; i < 8; ++i)
					cp_async16(tileB + b_sm + i * 8 * BN, gBt + b_gm + i * 8 * N);
				asm volatile("cp.async.commit_group;\n" ::);
				// 发了一枪预取，计数-1
				--nTiles;
				if (nTiles > 0)
					++tile; // 还有剩余 tile，计数+1
				pipe_w = pipe_r;
				pipe_r = (pipe_r == NPIPE - 1) ? 0 : pipe_r + 1;
			}
		}
	}

	asm volatile("cp.async.wait_all;\n" ::);
	asm volatile("cp.async.commit_group;\n" ::);
	__syncthreads();

	float *sC = reinterpret_cast<float *>(smem);
	#pragma unroll
	for (int i = 0; i < nM; ++i) {
		int row0 = lane / 4 + warp_row * 16 + i * 32;
		int row1 = row0 + 8;
		#pragma unroll
		for (int j = 0; j < nN; ++j) {
			int col = (lane % 4) * 2 + warp_col * 8 + j * 16;
			int idx = (i * nN + j) * 4;
			sC[row0 * BN + col + 0] = alpha * reg_c[idx + 0];
			sC[row0 * BN + col + 1] = alpha * reg_c[idx + 1];
			sC[row1 * BN + col + 0] = alpha * reg_c[idx + 2];
			sC[row1 * BN + col + 1] = alpha * reg_c[idx + 3];
		}
	}
	__syncthreads();

	int nVec = (BM * BN) / 4;
	int stride = BN / 4;
	for (int i = tid; i < nVec; i += NTHREADS) {
		int row = i / stride;
		int col = (i % stride) * 4;
		float4 v = reinterpret_cast<float4 *>(sC)[i];

		if (beta != 0.f) {
			float2 old = *reinterpret_cast<float2 *>(gC + row * N + col);
			half2 *oh = reinterpret_cast<half2 *>(&old);
			v.x += beta * __low2float(oh[0]);
			v.y += beta * __high2float(oh[0]);
			v.z += beta * __low2float(oh[1]);
			v.w += beta * __high2float(oh[1]);
		}

		half2 h0 = __floats2half2_rn(v.x, v.y);
		half2 h1 = __floats2half2_rn(v.z, v.w);
		half2 pack[2] = {h0, h1};
		*reinterpret_cast<float2 *>(gC + row * N + col) =
			*reinterpret_cast<float2 *>(pack);
	}
}

template <int BM, int BN, int NTHREADS>
__global__ void gemm_naive_kernel(
	const half *__restrict__ A,
	const half *__restrict__ B,
	half *__restrict__ C,
	int M, int N, int K,
	float alpha, float beta
) {
	constexpr int m_tid = 16;
	constexpr int n_tid = 8;
	constexpr int m_val = BM / m_tid;
	constexpr int n_val = BN / n_tid;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;

	int row0 = (tid / n_tid) * m_val;
	int col0 = (tid % n_tid) * n_val;

	const half *gA = A + bx * BM * K;
	const half *gB = B + by * BN;
	half *gC = C + bx * BM * N + by * BN;

	#pragma unroll
	for (int i = 0; i < m_val; ++i) {
		int lr = row0 + i;
		int gr = bx * BM + lr;
		if (gr >= M) continue;
		const half *a_row = gA + lr * K;

		#pragma unroll
		for (int j = 0; j < n_val; ++j) {
			int lc = col0 + j;
			int gc = by * BN + lc;
			if (gc >= N) continue;

			float acc = 0.f;
			const half *b_col = gB + lc;
			for (int k = 0; k < K; ++k)
				acc += __half2float(a_row[k]) * __half2float(b_col[k * N]);

			float old = (beta != 0.f) ? beta * __half2float(gC[lr * N + lc]) : 0.f;
			gC[lr * N + lc] = __float2half(alpha * acc + old);
		}
	}
}

extern "C" void solve(
	const half *A, const half *B, half *C,
	int M, int N, int K,
	float alpha, float beta
) {
	if (M == 1024 && N == 1024 && K == 1024) {
		constexpr int BM = 128;
		constexpr int BN = 128;
		constexpr int BK = 64;
		constexpr int NTHREADS = 128;
		constexpr int NPIPE = 3;

		dim3 block(NTHREADS);
		dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
		int smem = (int)(sizeof(half) * (BM + BN) * BK * NPIPE);

		cudaFuncSetAttribute(
			gemm_tc_kernel<BM, BN, BK, NTHREADS, NPIPE>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
		gemm_tc_kernel<BM, BN, BK, NTHREADS, NPIPE>
			<<<grid, block, smem>>>(A, B, C, M, N, K, alpha, beta);
	} else {
		constexpr int BM = 128;
		constexpr int BN = 64;
		constexpr int NTHREADS = 128;

		dim3 block(NTHREADS);
		dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
		gemm_naive_kernel<BM, BN, NTHREADS><<<grid, block>>>(
			A, B, C, M, N, K, alpha, beta);
	}
}
