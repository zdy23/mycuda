#include <cuda_runtime.h>
#include <stdint.h>

const int BLOCK_SIZE = 256;

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
// Each thread handles one element in its block's assigned range.
template <typename T>
__global__ void histogram_kernel(const T *__restrict__ input,
                                 uint32_t *__restrict__ block_hist, int n,
                                 int shift) {
  __shared__ uint32_t local_hist[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    local_hist[i] = 0;

  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    uint32_t bucket = get_bucket(input[idx], shift);
    atomicAdd(&local_hist[bucket], 1);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    block_hist[blockIdx.x * 256 + i] = local_hist[i];
}

// Block-level prefix scan: for each bucket, scan across all blocks
__global__ void block_scan_kernel(const uint32_t *__restrict__ d_block_hist,
                                  uint32_t *__restrict__ d_block_base,
                                  uint32_t *__restrict__ d_totals,
                                  int num_blocks) {
  int bucket = threadIdx.x;
  uint32_t sum = 0;
  for (int b = 0; b < num_blocks; b++) {
    d_block_base[b * 256 + bucket] = sum;
    sum += d_block_hist[b * 256 + bucket];
  }
  d_totals[bucket] = sum;
}

// Global prefix scan (exclusive) on 256-element totals
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

// Parallel scatter: all threads write, block-level prefix scan for offsets
template <typename T>
__global__ void
scatter_kernel(const T *__restrict__ input, T *__restrict__ output,
               const uint32_t *__restrict__ d_offsets,
               const uint32_t *__restrict__ d_block_base, int n, int shift) {
  __shared__ T local_in[256];
  __shared__ uint32_t local_bucket[256];
  __shared__ uint32_t hist[256];
  __shared__ uint32_t scan[256];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  if (gid < n) {
    T val = input[gid];
    local_in[tid] = val;
    local_bucket[tid] = get_bucket(val, shift);
  }

  for (int i = tid; i < 256; i += blockDim.x)
    hist[i] = 0;
  __syncthreads();

  if (gid < n)
    atomicAdd(&hist[local_bucket[tid]], 1);
  __syncthreads();

  scan[tid] = hist[tid];
  __syncthreads();

  for (int d = 1; d < 256; d <<= 1) {
    int idx = (tid + 1) * 2 * d - 1;
    if (idx < 256)
      scan[idx] += scan[idx - d];
    __syncthreads();
  }

  if (tid == 0)
    scan[255] = 0;
  __syncthreads();

  for (int d = 128; d > 0; d >>= 1) {
    int idx = (tid + 1) * 2 * d - 1;
    if (idx < 256) {
      uint32_t t = scan[idx];
      scan[idx] += scan[idx - d];
      scan[idx - d] = t;
    }
    __syncthreads();
  }

  hist[tid] = scan[tid];
  __syncthreads();

  if (gid < n) {
    uint32_t b = local_bucket[tid];
    uint32_t pos = atomicAdd(&hist[b], 1);
    output[d_offsets[b] + d_block_base[blockIdx.x * 256 + b] + pos] =
        local_in[tid];
  }
}

// Function to perform radix sort on the input data using CUDA
template <typename T> void radix_sort(T *d_input, T *d_output, int n) {
  constexpr int BITS_PER_PASS = 8;

  int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  uint32_t *d_block_hist, *d_block_base, *d_totals, *d_offsets;

  size_t block_array_size = (size_t)gridSize * 256 * sizeof(uint32_t);

  cudaMalloc(&d_block_hist, block_array_size);
  cudaMalloc(&d_block_base, block_array_size);
  cudaMalloc(&d_totals, 256 * sizeof(uint32_t));
  cudaMalloc(&d_offsets, 256 * sizeof(uint32_t));

  T *in = d_input;
  T *out = d_output;

  for (int shift = 0; shift < sizeof(T) * 8; shift += BITS_PER_PASS) {
    // 1. Per-block histogram (no global atomics)
    histogram_kernel<<<gridSize, BLOCK_SIZE>>>(in, d_block_hist, n, shift);

    // 2. Block-level scan: compute block_base and per-bucket totals
    block_scan_kernel<<<1, 256>>>(d_block_hist, d_block_base, d_totals,
                                  gridSize);

    // 3. Global prefix scan: compute offsets from totals
    prefix_scan_kernel<<<1, 256>>>(d_totals, d_offsets);

    // 4. Stable scatter using global_offsets + block_base
    scatter_kernel<<<gridSize, BLOCK_SIZE>>>(in, out, d_offsets, d_block_base,
                                             n, shift);

    std::swap(in, out);
  }

  if (in != d_output) {
    cudaMemcpy(d_output, in, n * sizeof(T), cudaMemcpyDeviceToDevice);
  }

  cudaFree(d_block_hist);
  cudaFree(d_block_base);
  cudaFree(d_totals);
  cudaFree(d_offsets);
}

// Explicit template instantiations
template void radix_sort<float>(float *, float *, int);
template void radix_sort<double>(double *, double *, int);

// ---- test harness -----------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T> bool verify(const T *data, int n) {
  for (int i = 0; i < n - 1; i++) {
    if (data[i] > data[i + 1])
      return false;
  }
  return true;
}

template <typename T> void run_test(int n) {
  T *h_input = new T[n];
  T *h_output = new T[n];
  T *h_expected = new T[n];

  // Generate random data including negatives
  srand(42);
  for (int i = 0; i < n; i++) {
    if constexpr (std::is_same_v<T, float>)
      h_input[i] = (float)(rand() % 1000000 - 500000) / 1000.0f;
    else
      h_input[i] = (double)(rand() % 1000000 - 500000) / 1000.0;
  }

  // CPU reference
  std::copy(h_input, h_input + n, h_expected);
  std::sort(h_expected, h_expected + n);

  // GPU sort
  T *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, n * sizeof(T), cudaMemcpyHostToDevice));

  // Warm up
  radix_sort<T>(d_input, d_output, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed run
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, n * sizeof(T), cudaMemcpyHostToDevice));
  auto start = std::chrono::high_resolution_clock::now();
  radix_sort<T>(d_input, d_output, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  CUDA_CHECK(
      cudaMemcpy(h_output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (h_output[i] != h_expected[i]) {
      ok = false;
      printf("MISMATCH at [%d]: got %g, expected %g\n", i, (double)h_output[i],
             (double)h_expected[i]);
      break;
    }
  }

  const char *tname = std::is_same_v<T, float> ? "float32" : "float64";
  printf("[%s] n=%d   %s   %.3f ms\n", tname, n, ok ? "PASS" : "FAIL", ms);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  delete[] h_input;
  delete[] h_output;
  delete[] h_expected;
}

// Quick debug: check first pass only
int main() {
  int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
  for (int n : sizes) {
    run_test<float>(n);
    run_test<double>(n);
  }
  printf("\nDone.\n");
  return 0;
}
