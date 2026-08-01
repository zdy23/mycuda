#!POPCORN leaderboard sort_v2
#!POPCORN gpu B200

from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/extension.h>

extern "C" {
void launch_sort_f32(float*, float*, int);
void launch_sort_f64(double*, double*, int);
}

torch::Tensor sort_f32(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty_like(input);
    launch_sort_f32(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

torch::Tensor sort_f64(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat64, "input must be float64");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty_like(input);
    launch_sort_f64(input.data_ptr<double>(), output.data_ptr<double>(), input.numel());
    return output;
}
"""

cuda_src = """
#include <cuda_runtime.h>
#include <cstdint>

// ---- key extraction ----
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

// ---- per-block histogram kernel ----
// Each block computes a local 256-bucket histogram for its chunk of input.
// Result stored in d_block_hists[blockIdx.x * 256 + bucket].
template <typename T>
__global__ void block_histogram_kernel(const T *__restrict__ input,
                                        uint32_t *__restrict__ block_hists,
                                        int n, int shift) {
  __shared__ uint32_t local_hist[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    local_hist[i] = 0;
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; idx < n; idx += stride) {
    uint32_t bucket = get_bucket(input[idx], shift);
    atomicAdd(&local_hist[bucket], 1);
  }
  __syncthreads();

  // Write block histogram to global memory (one writer per bucket)
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    block_hists[blockIdx.x * 256 + i] = local_hist[i];
}

// ---- block-level prefix scan ----
// For each bucket k (0..255), scan across all blocks to compute
// the exclusive prefix sum over block histograms.
// Also outputs the total count per bucket in d_totals.
// d_hists:  input block histograms (read-only)
// d_base:   output block base offsets (exclusive sum across blocks)
// d_totals: output per-bucket total count
__global__ void block_scan_kernel(const uint32_t *__restrict__ d_hists,
                                   uint32_t *__restrict__ d_base,
                                   uint32_t *__restrict__ d_totals,
                                   int num_blocks) {
  int bucket = threadIdx.x;  // 0..255
  uint32_t sum = 0;
  for (int b = 0; b < num_blocks; b++) {
    uint32_t cnt = d_hists[b * 256 + bucket];
    d_base[b * 256 + bucket] = sum;
    sum += cnt;
  }
  d_totals[bucket] = sum;
}

// ---- global prefix scan ----
// Computes exclusive prefix sum of a 256-element histogram.
// Input histogram[tid] = count for bucket tid.
// Output offsets[tid] = start position for bucket tid.
__global__ void prefix_scan_kernel(const uint32_t *__restrict__ histogram,
                                   uint32_t *__restrict__ offsets) {
  __shared__ uint32_t warp_sum[8];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  uint32_t val = histogram[tid];

#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset)
      val += y;
  }

  if (lane == 31)
    warp_sum[warp] = val;

  __syncthreads();

  if (warp == 0) {
    uint32_t x = (lane < 8) ? warp_sum[lane] : 0;

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

  if (warp > 0)
    val += warp_sum[warp - 1];

  offsets[tid] = val - histogram[tid];
}

// ---- stable scatter kernel ----
// Each block processes its chunk of elements in INPUT ORDER (sequentially
// by thread 0), guaranteeing stability.
// Output position = global_offsets[bucket] + block_base[bucket] + local_counter.
template <typename T>
__global__ void
stable_scatter_kernel(const T *__restrict__ input, T *__restrict__ output,
                      const uint32_t *__restrict__ global_offsets,
                      const uint32_t *__restrict__ block_base,
                      int n, int shift) {
  __shared__ uint32_t counters[256];
  __shared__ T local_in[256];

  // Load counters = global_offsets + block_base
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    counters[i] = global_offsets[i] + block_base[blockIdx.x * 256 + i];
  __syncthreads();

  // Each thread loads its element into shared memory at threadIdx.x
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    local_in[threadIdx.x] = input[idx];
  __syncthreads();

  // Thread 0 processes elements in input order (index 0, 1, 2, ...)
  if (threadIdx.x == 0) {
    int end = blockDim.x;
    int block_start = blockIdx.x * blockDim.x;
    if (block_start + end > n)
      end = n - block_start;
    for (int i = 0; i < end; i++) {
      T val = local_in[i];
      uint32_t bucket = get_bucket(val, shift);
      uint32_t pos = counters[bucket]++;
      output[pos] = val;
    }
  }
}

// ---- main radix sort ----
template <typename T>
void radix_sort(T *d_input, T *d_output, int n) {
  constexpr int BITS_PER_PASS = 8;
  constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  // Cap grid size to device maximum
  int maxGrid = 2147483647;
  cudaError_t err = cudaDeviceGetAttribute(&maxGrid, cudaDevAttrMaxGridDimX, 0);
  if (err != cudaSuccess) maxGrid = 2147483647;
  if (gridSize > maxGrid) gridSize = maxGrid;

  // Allocate per-block arrays and global histogram/offsets
  size_t block_array_size = (size_t)gridSize * NUM_BUCKETS * sizeof(uint32_t);
  uint32_t *d_block_hists;
  uint32_t *d_block_base;
  uint32_t *d_global_hist;
  uint32_t *d_global_offsets;

  cudaMalloc(&d_block_hists, block_array_size);
  cudaMalloc(&d_block_base, block_array_size);
  cudaMalloc(&d_global_hist, NUM_BUCKETS * sizeof(uint32_t));
  cudaMalloc(&d_global_offsets, NUM_BUCKETS * sizeof(uint32_t));

  T *in = d_input;
  T *out = d_output;

  for (int shift = 0; shift < sizeof(T) * 8; shift += BITS_PER_PASS) {
    // 1. Per-block histogram
    block_histogram_kernel<T><<<gridSize, blockSize>>>(in, d_block_hists, n, shift);

    // 2. Block-level scan: compute block_base and per-bucket totals
    block_scan_kernel<<<1, NUM_BUCKETS>>>(d_block_hists, d_block_base,
                                           d_global_hist, gridSize);

    // 3. Global prefix scan: compute global_offsets from global_hist
    prefix_scan_kernel<<<1, NUM_BUCKETS>>>(d_global_hist, d_global_offsets);

    // 4. Stable scatter using global_offsets + block_base
    stable_scatter_kernel<T><<<gridSize, blockSize>>>(in, out, d_global_offsets,
                                                       d_block_base, n, shift);

    T *tmp = in; in = out; out = tmp;
  }

  cudaFree(d_block_hists);
  cudaFree(d_block_base);
  cudaFree(d_global_hist);
  cudaFree(d_global_offsets);

  // If final result not in d_output, copy to d_output
  if (in != d_output) {
    cudaMemcpy(d_output, in, n * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

// ---- launch wrappers ----
extern "C" {
void launch_sort_f32(float *d_input, float *d_output, int n) {
    radix_sort<float>(d_input, d_output, n);
}
void launch_sort_f64(double *d_input, double *d_output, int n) {
    radix_sort<double>(d_input, d_output, n);
}
}

// ---- explicit template instantiations ----
template void radix_sort<float>(float*, float*, int);
template void radix_sort<double>(double*, double*, int);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="radix_sort_op",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["sort_f32", "sort_f64"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _module


def custom_kernel(data: input_t) -> output_t:
    # data is a tuple: (input_tensor,) or (input_tensor, output_tensor)
    if isinstance(data, tuple):
        input_tensor = data[0]
    else:
        input_tensor = data

    mod = _get_module()
    if input_tensor.dtype == torch.float32:
        result = mod.sort_f32(input_tensor.contiguous())
    elif input_tensor.dtype == torch.float64:
        result = mod.sort_f64(input_tensor.contiguous())
    else:
        raise TypeError(f"unsupported dtype: {input_tensor.dtype}")

    # If output buffer was provided in the tuple, copy result into it
    if isinstance(data, tuple) and len(data) > 1:
        data[1].copy_(result)
        return data[1]
    return result
