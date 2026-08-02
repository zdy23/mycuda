#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

const int blockSize = 256;

// Key extraction: bit-level bucket for radix sort
// float32: flip sign bit so negative values sort before positive
template <typename T>
__device__ __forceinline__ uint32_t get_bucket(T val, int shift);

template <>
__device__ __forceinline__ uint32_t get_bucket<float>(float val, int shift) {
  uint32_t bits = __float_as_uint(val);
  uint32_t mask = (int32_t(bits) >> 31) | 0x80000000u;
  return ((bits ^ mask) >> shift) & 0xFF;
}

template <>
__device__ __forceinline__ uint32_t get_bucket<double>(double val, int shift) {
  uint64_t bits = __double_as_longlong(val);
  uint64_t mask = (int64_t(bits) >> 63) | 0x8000000000000000ULL;
  return (uint32_t)((bits ^ mask) >> shift) & 0xFF;
}

// Kernel to compute the histogram of the input data based on the current shift
template <typename T>
__global__ void histogram_kernel(const T *__restrict__ input,
                                 uint32_t *__restrict__ block_hist, int n,
                                 int shift) {
  __shared__ uint32_t local_hist[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    local_hist[i] = 0;

  __syncthreads();

  constexpr int UNROLL = 4;

  int stride = blockDim.x * gridDim.x * UNROLL;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;

  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      int cur = idx + i;
      if (cur < n) {
        uint32_t bucket = get_bucket(input[cur], shift);
        atomicAdd(&local_hist[bucket], 1);
      }
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    block_hist[blockIdx.x * 256 + i] = local_hist[i];
}

__global__ void block_scan_kernel(const uint32_t *__restrict__ d_block_hist,
                                  uint32_t *__restrict__ d_block_base,
                                  uint32_t *__restrict__ d_totals,
                                  int num_blocks) {
  uint32_t sum = 0;

  const int tid = threadIdx.x;

  for (int b = 0; b < num_blocks; b++) {
    d_block_base[b * blockSize + tid] = sum;
    sum += d_block_hist[b * 256 + tid];
  }
  d_totals[tid] = sum;
}
// Kernel to perform prefix scan (exclusive) on the histogram to compute
// offsets
__global__ void prefix_scan_kernel(const uint32_t *__restrict__ d_totals,
                                   uint32_t *__restrict__ d_offsets) {
  constexpr int N = 256;

  __shared__ uint32_t warp_sum[8];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  uint32_t val = d_totals[tid];

// Perform warp-level scan using shuffle instructions
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset)
      val += y;
  }

  // Store the sum of each warp in shared memory
  if (lane == 31)
    warp_sum[warp] = val;

  __syncthreads();

  // Add the sum of previous warps to the current warp's value
  if (warp == 0) {
    uint32_t x = (lane < 8) ? warp_sum[lane] : 0;

// Scan within the first warp to compute the sum of the first 8 warps
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffff, x, offset);
      if (lane >= offset)
        x += y;
    }

    if (lane < 8)
      warp_sum[lane] = x;
  }

  __syncthreads();

  // Add the sum of previous warps to the current warp's value
  if (warp > 0)
    val += warp_sum[warp - 1];

  // Inclusive -> Exclusive
  d_offsets[tid] = val - d_totals[tid];
}

// Kernel to scatter the input data to the output based on the computed offsets
// Uses global atomicAdd on offsets to avoid cross-block bucket collisions.
// offsets is consumed (incremented) during scatter; recomputed each pass.
template <typename T>
__global__ void
scatter_kernel(const T *__restrict__ input, T *__restrict__ output,
               uint32_t *__restrict__ d_offsets, int n, int shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    uint32_t bucket = get_bucket(input[idx], shift);
    uint32_t pos = atomicAdd(&d_offsets[bucket], 1);
    output[pos] = input[idx];
  }
}

// Function to perform radix sort on the input data using CUDA
template <typename T> void radix_sort(T *d_input, T *d_output, int n) {
  constexpr int BITS_PER_PASS = 8;
  constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  uint32_t *d_block_hist, *d_block_base, *d_totals, *d_offsets;

  size_t block_array_size = (size_t)gridSize * 256 * sizeof(uint32_t);

  cudaMalloc(&d_block_hist, block_array_size);
  cudaMalloc(&d_block_base, block_array_size);
  cudaMalloc(&d_totals, 256 * sizeof(uint32_t));
  cudaMalloc(&d_offsets, 256 * sizeof(uint32_t));

  T *in = d_input;
  T *out = d_output;

  for (int shift = 0; shift < sizeof(T) * 8; shift += BITS_PER_PASS) {
    cudaMemset(d_block_hist, 0, block_array_size);

    histogram_kernel<<<gridSize, blockSize>>>(in, d_block_hist, n, shift);

    prefix_scan_kernel<<<1, NUM_BUCKETS>>>(d_block_hist, d_offsets);

    scatter_kernel<<<gridSize, blockSize>>>(in, out, d_offsets, n, shift);

    std::swap(in, out);
  }

  cudaFree(d_block_hist);
  cudaFree(d_block_base);
  cudaFree(d_totals);
  cudaFree(d_offsets);
}

// Explicit template instantiations
template void radix_sort<float>(float *, float *, int);
template void radix_sort<double>(double *, double *, int);
