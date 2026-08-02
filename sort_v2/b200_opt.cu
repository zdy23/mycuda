#include <cstdint>
#include <cuda.h>
#include <cuda/barrier>

const int BLOCK_SIZE = 256;
const int TILE_SIZE_32 = 2048;
const int TILE_SIZE_F64 = 1024;

// ---- Key extraction: bit-level bucket for radix sort ----
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

// ---- TMA tensor map helper ----
void make_tensor_map_1d(CUtensorMap *map, void *ptr, int n,
                        CUtensorMapDataType dtype, int tile_size) {
  uint64_t size[1] = {(uint64_t)n};
  uint64_t stride[1] = {1};
  uint32_t box_size[1] = {(uint32_t)tile_size};
  uint32_t elem_stride[1] = {1};
  cuTensorMapEncodeTiled(
      map, dtype, 1, ptr, size, stride, box_size, elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ---- TMA histogram kernel ----
// Loads TILE_SIZE elements via TMA, then each warp independently computes
// a histogram for its assigned sub-block (= BLOCK_SIZE elements).
// Outputs SUB_BLOCKS histogram rows per thread block, so total block_hist
// rows == gridSize (matching the scatter grid).
template <typename T, int TILE_SIZE>
__global__ void
histogram_kernel_tma(const __grid_constant__ CUtensorMap tensor_map,
                     uint32_t *__restrict__ block_hist, int n, int shift,
                     int total_blocks) {
  constexpr int ELEMS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  constexpr int SUB_BLOCKS = TILE_SIZE / BLOCK_SIZE;
  constexpr int WARPS_PER_SUB = 2048 / TILE_SIZE; // 1 (float) or 2 (double)

  __shared__ __align__(128) T smem_tile[TILE_SIZE];
  __shared__ uint32_t warp_hist[8][256];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  // Zero warp histograms
  for (int i = tid; i < 8 * 256; i += BLOCK_SIZE)
    ((uint32_t *)warp_hist)[i] = 0;

  // TMA load: one tile covers TILE_SIZE elements
  if (tid == 0) {
    init(&bar, 1);
    cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
        smem_tile, &tensor_map, (int)(blockIdx.x * TILE_SIZE), bar);
  }
  __syncthreads();
  bar.arrive_and_wait();

  int valid = min(TILE_SIZE, n - (int)(blockIdx.x * TILE_SIZE));

  // Each warp processes its stripe of the tile: contiguous elements within
  // one sub-block (no straddling).  warp w covers elements
  // [w*32*ELEMS_PER_THREAD, (w+1)*32*ELEMS_PER_THREAD).
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    int idx = warp * 32 * ELEMS_PER_THREAD + e * 32 + lane;
    if (idx < valid) {
      uint32_t bucket = get_bucket(smem_tile[idx], shift);
      atomicAdd(&warp_hist[warp][bucket], 1u);
    }
  }
  __syncthreads();

  // Combine warp histograms into sub-block histograms and write out.
  // Only write sub-blocks that map to valid logical blocks (< total_blocks).
  for (int i = tid; i < SUB_BLOCKS * 256; i += BLOCK_SIZE) {
    int sb = i / 256;
    int bucket = i % 256;
    int global_sb = blockIdx.x * SUB_BLOCKS + sb;
    if (global_sb < total_blocks) {
      uint32_t sum = 0;
#pragma unroll
      for (int w = 0; w < WARPS_PER_SUB; w++)
        sum += warp_hist[sb * WARPS_PER_SUB + w][bucket];
      block_hist[global_sb * 256 + bucket] = sum;
    }
  }
}

// ---- Block-level scan ----
// For each bucket (0..255), exclusive scan across all logical blocks.
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

// ---- Global prefix scan on 256-element totals ----
// Warp-level Kogge-Stone + single-warp reduction.
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

// ---- TMA scatter kernel ----
// TMA loads one BLOCK_SIZE tile into shared memory; scatter logic mirrors
// the original stable scatter (per-warp counters + intra-warp shfl rank).
template <typename T>
__global__ void scatter_kernel_tma(
    const __grid_constant__ CUtensorMap tensor_map, T *__restrict__ output,
    const uint32_t *__restrict__ d_offsets,
    const uint32_t *__restrict__ d_block_base, int n, int shift) {
  __shared__ __align__(128) T smem_tile[BLOCK_SIZE];
  __shared__ uint32_t local_bucket[256];
  __shared__ uint32_t warp_count[8][256];
  __shared__ uint32_t warp_offs[8][256];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int warp = tid >> 5;
  int lane = tid & 31;

  for (int i = tid; i < 256; i += blockDim.x)
    local_bucket[i] = 0xFFFFFFFF;
  __syncthreads();

  // TMA load — replaces per-thread global reads
  if (tid == 0) {
    init(&bar, 1);
    cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
        smem_tile, &tensor_map, (int)(blockIdx.x * BLOCK_SIZE), bar);
  }
  __syncthreads();
  bar.arrive_and_wait();

  if (gid < n) {
    local_bucket[tid] = get_bucket(smem_tile[tid], shift);
  }

  for (int i = tid; i < 8 * 256; i += blockDim.x)
    ((uint32_t *)warp_count)[i] = 0;
  __syncthreads();

  if (gid < n)
    atomicAdd(&warp_count[warp][local_bucket[tid]], 1);
  __syncthreads();

  for (int b = tid; b < 256; b += blockDim.x) {
    uint32_t sum = 0;
    for (int w = 0; w < 8; w++) {
      warp_offs[w][b] = sum;
      sum += warp_count[w][b];
    }
  }
  __syncthreads();

  uint32_t b = local_bucket[tid];
  uint32_t rank = 0;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    uint32_t other = __shfl_sync(0xffffffff, b, i);
    rank += (i < lane && other == b);
  }

  if (gid < n) {
    output[d_offsets[b] + d_block_base[blockIdx.x * 256 + b] +
           warp_offs[warp][b] + rank] = smem_tile[tid];
  }
}

// ---- TMA-optimized radix sort ----
template <typename T> void radix_sort_tma(T *d_input, T *d_output, int n) {
  constexpr int BITS_PER_PASS = 8;
  constexpr int TILE_SIZE =
      std::is_same_v<T, float> ? TILE_SIZE_32 : TILE_SIZE_F64;
  constexpr int SUB_BLOCKS = TILE_SIZE / BLOCK_SIZE;

  int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int histGrid = (n + TILE_SIZE - 1) / TILE_SIZE;
  // histGrid * SUB_BLOCKS >= gridSize; extra rows (beyond gridSize) are
  // zero-filled by histogram kernel and skipped by block_scan.
  int allocBlocks = histGrid * SUB_BLOCKS;

  static uint32_t *d_block_hist = nullptr;
  static uint32_t *d_block_base = nullptr;
  static uint32_t *d_totals = nullptr;
  static uint32_t *d_offsets = nullptr;
  static int last_n = 0;

  if (n > last_n) {
    if (d_block_hist)
      cudaFree(d_block_hist);
    if (d_block_base)
      cudaFree(d_block_base);
    size_t block_array_size = (size_t)allocBlocks * 256 * sizeof(uint32_t);
    cudaMalloc(&d_block_hist, block_array_size);
    cudaMalloc(&d_block_base, block_array_size);
    last_n = n;
  }

  if (!d_totals) {
    cudaMalloc(&d_totals, 256 * sizeof(uint32_t));
    cudaMalloc(&d_offsets, 256 * sizeof(uint32_t));
  }

  T *in = d_input;
  T *out = d_output;

  CUtensorMapDataType dtype = std::is_same_v<T, float>
                                  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                                  : CU_TENSOR_MAP_DATA_TYPE_FLOAT64;

  for (int shift = 0; shift < sizeof(T) * 8; shift += BITS_PER_PASS) {
    // 1. Per-block histogram via TMA (large tiles for bandwidth)
    {
      CUtensorMap tmap_hist;
      make_tensor_map_1d(&tmap_hist, in, n, dtype, TILE_SIZE);
      histogram_kernel_tma<T, TILE_SIZE><<<histGrid, BLOCK_SIZE>>>(
          tmap_hist, d_block_hist, n, shift, gridSize);
    }

    // 2. Block-level exclusive scan
    block_scan_kernel<<<1, 256>>>(d_block_hist, d_block_base, d_totals,
                                  gridSize);

    // 3. Global prefix scan
    prefix_scan_kernel<<<1, 256>>>(d_totals, d_offsets);

    // 4. Stable scatter via TMA load
    {
      CUtensorMap tmap_scatter;
      make_tensor_map_1d(&tmap_scatter, in, n, dtype, BLOCK_SIZE);
      scatter_kernel_tma<T><<<gridSize, BLOCK_SIZE>>>(
          tmap_scatter, out, d_offsets, d_block_base, n, shift);
    }

    std::swap(in, out);
  }

  // Final copy if output is in the wrong buffer
  if (in != d_output) {
    cudaMemcpy(d_output, in, n * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

// Explicit instantiations
template void radix_sort_tma<float>(float *, float *, int);
template void radix_sort_tma<double>(double *, double *, int);

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

template <typename T> void run_test(int n) {
  T *h_input = new T[n];
  T *h_output = new T[n];
  T *h_expected = new T[n];

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

  T *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, n * sizeof(T), cudaMemcpyHostToDevice));

  // Warm up
  radix_sort_tma<T>(d_input, d_output, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed run
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, n * sizeof(T), cudaMemcpyHostToDevice));
  auto start = std::chrono::high_resolution_clock::now();
  radix_sort_tma<T>(d_input, d_output, n);
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
  printf("[TMA %s] n=%d   %s   %.3f ms\n", tname, n, ok ? "PASS" : "FAIL", ms);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  delete[] h_input;
  delete[] h_output;
  delete[] h_expected;
}

int main() {
  int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
  for (int n : sizes) {
    run_test<float>(n);
    run_test<double>(n);
  }
  printf("\nDone.\n");
  return 0;
}
