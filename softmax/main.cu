#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

// How many float4s each thread loads before computing (memory-level parallelism).
#define UNROLL 4

// online softmax for 32 lanes, into one
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

// Streaming load/store: bypass L2 (data is touched once).
#ifdef __clang__
#define LDCS(ptr) (*(ptr))
#define STCS(ptr, val) (*(ptr) = (val))
#else
#define LDCS(ptr) __ldcs(ptr)
#define STCS(ptr, val) __stcs(ptr, val)
#endif

// Fold one element into the running (max, sum) bucket (online softmax).
__device__ __forceinline__ void online_update(float val, float& m, float& s) {
	float prev = m;
	m = fmaxf(prev, val);
	s = s * __expf(prev - m) + __expf(val - m);
}

// Per-block partial (max, sum), exchanged between pass1 and pass2.
// 2048 slots: safe even for 256-thread blocks (8 blocks/SM x 108 SMs on A100).
__device__ float sumvals[2048];
__device__ float maxvals[2048];

// maxval and sumval in each block
template <int threadsPerBlock>
__global__ void __launch_bounds__(threadsPerBlock) softmax_kernel_pass1(const float* input, int N) {
	__shared__ float smem[(threadsPerBlock) / 32 * 2];

	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + tid;
	int stride = gridDim.x * blockDim.x;
	int step = stride * UNROLL; // one UNROLL-chunk of float4s
	int n4 = N >> 2; // number of float4 chunks
	const float4* vec = reinterpret_cast<const float4*>(input);

	float warp_max = -FLT_MAX;
	float warp_sum = 0.0f;

	// Main loop: issue UNROLL independent float4 loads, then compute.
	// Overlaps memory latency across the batch of loads.
	int i = id;
	for (; i + (UNROLL - 1) * stride < n4; i += step) {
		float4 regs[UNROLL];
		#pragma unroll
		for (int u = 0; u < UNROLL; u++)
			regs[u] = LDCS(&vec[i + u * stride]);
		#pragma unroll
		for (int u = 0; u < UNROLL; u++) {
			online_update(regs[u].x, warp_max, warp_sum);
			online_update(regs[u].y, warp_max, warp_sum);
			online_update(regs[u].z, warp_max, warp_sum);
			online_update(regs[u].w, warp_max, warp_sum);
		}
	}
	// Leftover float4s that don't fill a full UNROLL batch.
	for (; i < n4; i += stride) {
		float4 v = LDCS(&vec[i]);
		online_update(v.x, warp_max, warp_sum);
		online_update(v.y, warp_max, warp_sum);
		online_update(v.z, warp_max, warp_sum);
		online_update(v.w, warp_max, warp_sum);
	}
	// Scalar tail: N % 4 leftovers.
	for (int j = (n4 << 2) + id; j < N; j += stride)
		online_update(LDCS(&input[j]), warp_max, warp_sum);

	// warp reduction
	warp_reduce_online_softmax(warp_sum, warp_max);

	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = threadsPerBlock / 32;

	if (lane_id == 0) {
		smem[warp_id] = warp_sum;
		smem[warp_id + num_warps] = warp_max;
	}
	__syncthreads();

	// load blocks' data into a warp and use global array to store.
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

template <int threadsPerBlock>
__global__ void __launch_bounds__(threadsPerBlock) softmax_kernel_pass2(const float* input, float* output, int N, int num_blocks) {
	__shared__ float final_max, final_sum;

	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + tid;

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

		warp_reduce_online_softmax(block_sum, block_max);
		if (tid == 0) {
			final_max = block_max;
			final_sum = block_sum;
		}
	}
	__syncthreads();

	int stride = gridDim.x * blockDim.x;
	int step = stride * UNROLL;
	int n4 = N >> 2;
	const float4* vec_in = reinterpret_cast<const float4*>(input);
	float4* vec_out = reinterpret_cast<float4*>(output);
	// Multiply by reciprocal: cheaper than per-element division.
	const float inv_sum = 1.0f / final_sum;

	// Main loop: UNROLL loads -> exp/normalize -> streaming stores.
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
			// Streaming store: output is write-once, no reuse -> bypass L2.
			STCS(&vec_out[i + u * stride], r);
		}
	}
	// Leftover float4s.
	for (; i < n4; i += stride) {
		float4 v = LDCS(&vec_in[i]);
		float4 r;
		r.x = __expf(v.x - final_max) * inv_sum;
		r.y = __expf(v.y - final_max) * inv_sum;
		r.z = __expf(v.z - final_max) * inv_sum;
		r.w = __expf(v.w - final_max) * inv_sum;
		STCS(&vec_out[i], r);
	}
	// Scalar tail.
	for (int j = (n4 << 2) + id; j < N; j += stride)
		STCS(&output[j], __expf(LDCS(&input[j]) - final_max) * inv_sum);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
	if (N <= 0)
		return;

	const int threadsPerBlock = 512;

	// Query device once, then cache: occupancy-based grid size.
	// Exactly one resident wave of blocks -> no tail wave, no oversubscription.
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
