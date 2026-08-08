#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

#define UNROLL 4

__device__ __forceinline__ void warp_reduce_online_softmax(float& warp_sum, float& warp_max) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float peer_sum = __shfl_down_sync(0xffffffff, warp_sum, offset);
        float peer_max = __shfl_down_sync(0xffffffff, warp_max, offset);

        float new_max = fmaxf(peer_max, warp_max);

        warp_sum = warp_sum * __expf(warp_max - new_max)
                 + peer_sum * __expf(peer_max - new_max);
        warp_max = new_max;
    }
}

#ifdef __clang__
#define LDCS(ptr) (*(ptr))
#define STCS(ptr, val) (*(ptr) = (val))
#else
#define LDCS(ptr) __ldcs(ptr)
#define STCS(ptr, val) __stcs(ptr, val)
#endif

__device__ __forceinline__ void online_update(float val, float& m, float& s) {
	float prev = m;
	m = fmaxf(prev, val);
	s = s * __expf(prev - m) + __expf(val - m);
}

// 记录每个线程块的最大值和总和，用于后续归一化
__device__ float sumvals[2048];
__device__ float maxvals[2048];

// 归约阶段，计算每个线程块的最大值和总和

template <int threadsPerBlock>
__global__ void __launch_bounds__(threadsPerBlock) softmax_kernel_pass1(const float* input, int N) {
	__shared__ float smem[(threadsPerBlock) / 32 * 2];

	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + tid;
	int stride = gridDim.x * blockDim.x;
	int step = stride * UNROLL;
	int n4 = N >> 2;

	// 用 float4 进行向量化加载和存储，以提高内存访问效率
	const float4* vec = reinterpret_cast<const float4*>(input);

	float warp_max = -FLT_MAX;
	float warp_sum = 0.0f;

	int i = id;
	for (; i + (UNROLL - 1) * stride < n4; i += step) {
		// 使用 float4 进行向量化加载
		float4 regs[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			regs[u] = LDCS(&vec[i + u * stride]);
		// 在线更新最大值和总和
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			online_update(regs[u].x, warp_max, warp_sum);
			online_update(regs[u].y, warp_max, warp_sum);
			online_update(regs[u].z, warp_max, warp_sum);
			online_update(regs[u].w, warp_max, warp_sum);
		}
	}

	// 处理剩余的元素
	for (; i < n4; i += stride) {
		float4 v = LDCS(&vec[i]);
		online_update(v.x, warp_max, warp_sum);
		online_update(v.y, warp_max, warp_sum);
		online_update(v.z, warp_max, warp_sum);
		online_update(v.w, warp_max, warp_sum);
	}

	// 处理剩余的元素（不足一个 float4 的部分）
	for (int j = (n4 << 2) + id; j < N; j += stride)
		online_update(LDCS(&input[j]), warp_max, warp_sum);

	// 进行 warp 内归约，计算每个 warp 的最大值和总和
	warp_reduce_online_softmax(warp_sum, warp_max);

	// 将每个 warp 的结果写入共享内存，以便后续归约
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = threadsPerBlock / 32;

	if (lane_id == 0) {
		smem[warp_id] = warp_sum;
		smem[warp_id + num_warps] = warp_max;
	}
	__syncthreads();

	// 最后一个 warp 负责将所有 warp 的结果归约为整个线程块的最大值和总和，并写入全局内存
	if (warp_id == 0) {
		float block_sum = (tid < num_warps) ? smem[tid] : 0.0f;
		float block_max = (tid < num_warps) ? smem[tid + num_warps] : -FLT_MAX;
		warp_reduce_online_softmax(block_sum, block_max);

		if (tid == 0) {
			maxvals[blockIdx.x] = block_max;
			sumvals[blockIdx.x] = block_sum;
		}
	}
}

// 归一化阶段，使用每个线程块的最大值和总和对输入进行归一化，计算 softmax 输出

template <int threadsPerBlock>
__global__ void __launch_bounds__(threadsPerBlock) softmax_kernel_pass2(const float* input, float* output, int N, int num_blocks) {
	__shared__ float final_max, final_sum;

	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + tid;

	// 归约所有线程块的最大值和总和，得到最终的最大值和总和
	if (tid < 32) {
		float block_max = -FLT_MAX;
		float block_sum = 0.0f;

		for (int i = threadIdx.x; i < num_blocks; i += 32) {
			float maxval = maxvals[i];
			float sumval = sumvals[i];
			float prev_max = block_max;
			block_max = fmaxf(prev_max, maxval);
			block_sum = block_sum * __expf(prev_max - block_max)
				+  sumval * __expf(maxval - block_max);
		}

		// 进行 warp 内归约，得到最终的最大值和总和
		warp_reduce_online_softmax(block_sum, block_max);
		if (tid == 0) {
			final_max = block_max;
			final_sum = block_sum;
		}
	}
	__syncthreads();

	// 使用最终的最大值和总和对输入进行归一化，计算 softmax 输出
	int stride = gridDim.x * blockDim.x;
	int step = stride * UNROLL;
	int n4 = N >> 2;
	const float4* vec_in = reinterpret_cast<const float4*>(input);
	float4* vec_out = reinterpret_cast<float4*>(output);
	const float inv_sum = 1.0f / final_sum;

	// 使用 float4 进行向量化加载和存储，以提高内存访问效率
	int i = id;
	for (; i + (UNROLL - 1) * stride < n4; i += step) {
		float4 regs[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			regs[u] = LDCS(&vec_in[i + u * stride]);
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			float4 r;
			r.x = __expf(regs[u].x - final_max) * inv_sum;
			r.y = __expf(regs[u].y - final_max) * inv_sum;
			r.z = __expf(regs[u].z - final_max) * inv_sum;
			r.w = __expf(regs[u].w - final_max) * inv_sum;
			STCS(&vec_out[i + u * stride], r);
		}
	}
	for (; i < n4; i += stride) {
		float4 v = LDCS(&vec_in[i]);
		float4 r;
		r.x = __expf(v.x - final_max) * inv_sum;
		r.y = __expf(v.y - final_max) * inv_sum;
		r.z = __expf(v.z - final_max) * inv_sum;
		r.w = __expf(v.w - final_max) * inv_sum;
		STCS(&vec_out[i], r);
	}
	for (int j = (n4 << 2) + id; j < N; j += stride)
		STCS(&output[j], __expf(LDCS(&input[j]) - final_max) * inv_sum);
}

extern "C" void solve(const float* input, float* output, int N) {
	if (N <= 0)
		return;

	const int threadsPerBlock = 512;

	static int maxBlocks = -1;
	if (maxBlocks < 0) {
		int nb_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&nb_per_sm, softmax_kernel_pass1<threadsPerBlock>, threadsPerBlock, 0);
		int sm_count = 0;
		cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
		maxBlocks = nb_per_sm * sm_count;
		if (maxBlocks < 1) maxBlocks = 1;
	}

	int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);

	softmax_kernel_pass1<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(input, N);
	softmax_kernel_pass2<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(input, output, N, blocksPerGrid);

	cudaDeviceSynchronize();
}
