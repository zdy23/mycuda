#!POPCORN leaderboard sort_v2
#!POPCORN gpu B200

"""
B200 sort_v2 submission.

Avoid torch.utils.cpp_extension.load_inline for CUDA: on the eval image
(nvcc host-compiles .cu with torch -isystem + -std=c++17) that path hits a
known ATen List_inl.h / dependent-scope error. Build a standalone .so with
raw nvcc (no torch headers) and call it via ctypes.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from task import input_t, output_t

# Pure CUDA — no torch, no TMA experimental APIs, no libcuda.
CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// ---- Key extraction (float/double total order via sign-bit flip) ----
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

template <typename T>
__global__ void
scatter_kernel(const T *__restrict__ input, T *__restrict__ output,
               const uint32_t *__restrict__ d_offsets,
               const uint32_t *__restrict__ d_block_base, int n, int shift) {
  __shared__ T local_in[256];
  __shared__ uint32_t local_bucket[256];
  __shared__ uint32_t warp_count[8][256];
  __shared__ uint32_t warp_offs[8][256];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int warp = tid >> 5;
  int lane = tid & 31;

  for (int i = tid; i < 256; i += blockDim.x)
    local_bucket[i] = 0xFFFFFFFFu;
  __syncthreads();

  if (gid < n) {
    local_in[tid] = input[gid];
    local_bucket[tid] = get_bucket(input[gid], shift);
  }

  for (int i = tid; i < 8 * 256; i += blockDim.x)
    ((uint32_t *)warp_count)[i] = 0;
  __syncthreads();

  if (gid < n)
    atomicAdd(&warp_count[warp][local_bucket[tid]], 1u);
  __syncthreads();

  for (int b = tid; b < 256; b += blockDim.x) {
    uint32_t sum = 0;
    for (int w = 0; w < 8; w++) {
      warp_offs[w][b] = sum;
      sum += warp_count[w][b];
    }
  }
  __syncthreads();

  // All lanes must execute shfl; only valid threads store.
  uint32_t b = local_bucket[tid];
  uint32_t rank = 0;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    uint32_t other = __shfl_sync(0xffffffff, b, i);
    rank += (i < lane && other == b);
  }

  if (gid < n) {
    output[d_offsets[b] + d_block_base[blockIdx.x * 256 + b] +
           warp_offs[warp][b] + rank] = local_in[tid];
  }
}

// Does not write d_input (uses scratch for ping-pong).
template <typename T>
void radix_sort(const T *d_input, T *d_output, int n) {
  if (n <= 0)
    return;

  constexpr int BITS_PER_PASS = 8;
  constexpr int NUM_PASSES = (int)(sizeof(T) * 8 / BITS_PER_PASS);

  int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  static uint32_t *d_block_hist = nullptr;
  static uint32_t *d_block_base = nullptr;
  static uint32_t *d_totals = nullptr;
  static uint32_t *d_offsets = nullptr;
  static T *d_scratch = nullptr;
  static int hist_cap = 0;
  static int scratch_cap = 0;

  if (gridSize > hist_cap) {
    if (d_block_hist)
      CUDA_CHECK(cudaFree(d_block_hist));
    if (d_block_base)
      CUDA_CHECK(cudaFree(d_block_base));
    size_t bytes = (size_t)gridSize * 256 * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_block_hist, bytes));
    CUDA_CHECK(cudaMalloc(&d_block_base, bytes));
    hist_cap = gridSize;
  }

  if (n > scratch_cap) {
    if (d_scratch)
      CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)n * sizeof(T)));
    scratch_cap = n;
  }

  if (!d_totals) {
    CUDA_CHECK(cudaMalloc(&d_totals, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, 256 * sizeof(uint32_t)));
  }

  // Pass 0: d_input -> d_output; then ping-pong output <-> scratch.
  const T *in = d_input;
  T *out = d_output;

  for (int pass = 0; pass < NUM_PASSES; pass++) {
    int shift = pass * BITS_PER_PASS;

    histogram_kernel<T>
        <<<gridSize, BLOCK_SIZE>>>(in, d_block_hist, n, shift);
    block_scan_kernel<<<1, 256>>>(d_block_hist, d_block_base, d_totals,
                                  gridSize);
    prefix_scan_kernel<<<1, 256>>>(d_totals, d_offsets);
    scatter_kernel<T><<<gridSize, BLOCK_SIZE>>>(in, out, d_offsets,
                                                d_block_base, n, shift);

    in = out;
    out = (out == d_output) ? d_scratch : d_output;
  }

  if (in != d_output) {
    CUDA_CHECK(cudaMemcpy(d_output, in, (size_t)n * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }
}

extern "C" {

void launch_sort_f32(const float *in, float *out, int n) {
  radix_sort<float>(in, out, n);
}

void launch_sort_f64(const double *in, double *out, int n) {
  radix_sort<double>(in, out, n);
}

}  // extern "C"
"""

_lib = None


def _cache_dir() -> Path:
    # Prefer torch extension cache if available; else /tmp.
    try:
        from torch.utils.cpp_extension import _get_build_directory

        d = Path(_get_build_directory("sort_v2_standalone_b200_v4", verbose=False))
    except Exception:
        d = Path(tempfile.gettempdir()) / "sort_v2_standalone_b200_v4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_nvcc() -> str:
    for key in ("CUDACXX", "NVCC"):
        p = os.environ.get(key)
        if p and os.path.isfile(p):
            return p
    for candidate in (
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ):
        if os.path.isfile(candidate):
            return candidate
    return "nvcc"


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    # Ensure CUDA context exists before we load a CUDA .so.
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")

    src_hash = hashlib.sha1(CUDA_SRC.encode("utf-8")).hexdigest()[:12]
    build = _cache_dir()
    cu_path = build / f"sort_{src_hash}.cu"
    so_path = build / f"libsort_{src_hash}.so"

    if not so_path.is_file():
        cu_path.write_text(CUDA_SRC)
        nvcc = _find_nvcc()
        cmd = [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-O3",
            "-std=c++17",
            # B200 = sm_100
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_100,code=compute_100",
            str(cu_path),
            "-o",
            str(so_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "nvcc failed building sort kernel.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}"
            ) from e

    lib = ctypes.CDLL(str(so_path))
    lib.launch_sort_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.launch_sort_f32.restype = None
    lib.launch_sort_f64.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.launch_sort_f64.restype = None
    _lib = lib
    return _lib


def custom_kernel(data: input_t) -> output_t:
    if isinstance(data, tuple):
        input_tensor = data[0]
    else:
        input_tensor = data

    x = input_tensor.contiguous()
    if not x.is_cuda:
        raise TypeError("input must be a CUDA tensor")

    n = int(x.numel())
    out = torch.empty_like(x)
    lib = _get_lib()

    if x.dtype == torch.float32:
        lib.launch_sort_f32(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            n,
        )
    elif x.dtype == torch.float64:
        lib.launch_sort_f64(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            n,
        )
    else:
        raise TypeError(f"unsupported dtype: {x.dtype}")

    # Synchronize so correctness checks see finished GPU work.
    torch.cuda.synchronize()

    if isinstance(data, tuple) and len(data) > 1:
        data[1].copy_(out)
        return data[1]
    return out
