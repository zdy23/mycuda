from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/extension.h>

void launch_sort_f32(float*, float*, int);
void launch_sort_f64(double*, double*, int);

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

// ---- histogram kernel ----
template <typename T>
__global__ void histogram_kernel(const T *__restrict__ input,
                                 uint32_t *__restrict__ histogram, int n,
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
    atomicAdd(&histogram[i], local_hist[i]);
}

// ---- prefix scan kernel ----
__global__ void prefix_scan_kernel(const uint32_t *__restrict__ histogram,
                                   uint32_t *__restrict__ offsets) {
  constexpr int N = 256;

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

// ---- scatter kernel ----
template <typename T>
__global__ void
scatter_kernel(const T *__restrict__ input, T *__restrict__ output,
               uint32_t *__restrict__ offsets, int n, int shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    uint32_t bucket = get_bucket(input[idx], shift);
    uint32_t pos = atomicAdd(&offsets[bucket], 1);
    output[pos] = input[idx];
  }
}

// ---- main radix sort ----
template <typename T>
void radix_sort(T *d_input, T *d_output, int n) {
  constexpr int BITS_PER_PASS = 8;
  constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  int maxGrid;
  cudaDeviceGetAttribute(&maxGrid, cudaDevAttrMaxGridDimX, 0);
  if (gridSize > maxGrid) gridSize = maxGrid;

  uint32_t *d_histogram;
  uint32_t *d_offsets;

  cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(uint32_t));
  cudaMalloc(&d_offsets, NUM_BUCKETS * sizeof(uint32_t));

  T *in = d_input;
  T *out = d_output;

  for (int shift = 0; shift < sizeof(T) * 8; shift += BITS_PER_PASS) {
    cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(uint32_t));

    histogram_kernel<T><<<gridSize, blockSize>>>(in, d_histogram, n, shift);
    prefix_scan_kernel<<<1, NUM_BUCKETS>>>(d_histogram, d_offsets);
    scatter_kernel<T><<<gridSize, blockSize>>>(in, out, d_offsets, n, shift);

    T *tmp = in; in = out; out = tmp;
  }

  cudaFree(d_histogram);
  cudaFree(d_offsets);

  // if final result not in d_input, copy back
  if (in != d_input) {
    cudaMemcpy(d_input, in, n * sizeof(T), cudaMemcpyDeviceToDevice);
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
    mod = _get_module()
    if data.dtype == torch.float32:
        return mod.sort_f32(data.contiguous())
    elif data.dtype == torch.float64:
        return mod.sort_f64(data.contiguous())
    else:
        raise TypeError(f"unsupported dtype: {data.dtype}")
