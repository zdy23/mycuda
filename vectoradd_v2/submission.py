#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

import torch
from torch.utils.cpp_extension import load_inline

try:
    from task import input_t, output_t
except ImportError:
    from typing import Any
    input_t = Any
    output_t = Any

cuda_source = """
#include <cuda_fp16.h>

/*
    CUDA kernels for vector addition with type-specialized vectorized paths.

    - float:  float4 vectorization (128-bit loads/stores), 2x ILP per iteration.
    - half:   uint4-based half8 vectorization (128-bit loads/stores via union),
              2x ILP per iteration.
    - Fallback: generic scalar kernel for double or unaligned float inputs.
*/

// Generic vector addition kernel for any scalar type
template <typename scalar_t>
__global__ __launch_bounds__(256, 8) void vector_add_kernel_generic(
    const scalar_t *__restrict__ A, const scalar_t *__restrict__ B,
    scalar_t *__restrict__ C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int i = idx; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

// float4 vectorized kernel: 2 float4 per iteration (ILP x2) to hide memory latency
__global__ __launch_bounds__(256, 8) void vector_add_kernel_float4(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int N) {
  const float4 *A4 = reinterpret_cast<const float4 *>(A);
  const float4 *B4 = reinterpret_cast<const float4 *>(B);
  float4 *C4 = reinterpret_cast<float4 *>(C);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int N4 = N / 4;

    // Process two float4 elements per iteration (i and i+stride) for ILP x2
    #pragma unroll
    for (int i = idx; i < N4; i += stride * 2) {
      // float4 at position i
      float4 a0 = A4[i];
      float4 b0 = B4[i];
      float4 c0;
      c0.x = a0.x + b0.x;
      c0.y = a0.y + b0.y;
      c0.z = a0.z + b0.z;
      c0.w = a0.w + b0.w;
      C4[i] = c0;

      // float4 at position i + stride
      int i2 = i + stride;
      if (i2 < N4) {
        float4 a1 = A4[i2];
        float4 b1 = B4[i2];
        float4 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;
        C4[i2] = c1;
      }
    }

    // Scalar tail for remaining floats
    int tail = N4 * 4;
  for (int i = tail + idx; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

// half8 vectorized kernel: 2 half8 per iteration (ILP x2) to hide memory latency
__global__ __launch_bounds__(256, 8) void vector_add_kernel_half8(
    const half *__restrict__ A, const half *__restrict__ B,
    half *__restrict__ C, int N) {
  const uint4 *A8 = reinterpret_cast<const uint4 *>(A);
  const uint4 *B8 = reinterpret_cast<const uint4 *>(B);
  uint4 *C8 = reinterpret_cast<uint4 *>(C);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int N8 = N / 8;

  // Process two half8 elements per iteration (i and i+stride) for ILP x2
#pragma unroll
  for (int i = idx; i < N8; i += stride * 2) {
    union {
      uint4 u4;
      half2 h2s[4];
    } a0, b0, c0;
    a0.u4 = A8[i];
    b0.u4 = B8[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      c0.h2s[j] = __hadd2(a0.h2s[j], b0.h2s[j]);
    }
    C8[i] = c0.u4;

    int i2 = i + stride;
    if (i2 < N8) {
      union {
        uint4 u4;
        half2 h2s[4];
      } a1, b1, c1;
      a1.u4 = A8[i2];
      b1.u4 = B8[i2];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        c1.h2s[j] = __hadd2(a1.h2s[j], b1.h2s[j]);
      }
      C8[i2] = c1.u4;
    }
  }

  // half2 tail for remaining even halfs
  const half2 *A2 = reinterpret_cast<const half2 *>(A);
  const half2 *B2 = reinterpret_cast<const half2 *>(B);
  half2 *C2 = reinterpret_cast<half2 *>(C);


  int tail_half2 = N8 * 4;
  for (int i = tail_half2 + idx; i < N / 2; i += stride) {
    C2[i] = __hadd2(A2[i], B2[i]);
  }

  // Scalar half tail for final odd element
  int tail_half = (N / 2) * 2;
  for (int i = tail_half + idx; i < N; i += stride) {
    C[i] = __hadd(A[i], B[i]);
  }
}

// CUDA kernel launcher
torch::Tensor vector_add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.numel();

    const int threads = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Compute block count bounded by GPU maximum concurrent capacity
    int blocks = std::min(
        (N + threads - 1) / threads,
        prop.multiProcessorCount * 8
    );

    if (A.scalar_type() == torch::kHalf) {
        // Use half8 vectorized kernel for half precision inputs
        vector_add_kernel_half8<<<blocks, threads>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N
        );
    } else if (A.scalar_type() == torch::kFloat) {
        // Use float4 vectorized kernel when all addresses are 16-byte aligned,
        // otherwise fall back to the generic scalar kernel
        uintptr_t pa = reinterpret_cast<uintptr_t>(A.data_ptr<float>());
        uintptr_t pb = reinterpret_cast<uintptr_t>(B.data_ptr<float>());
        uintptr_t pc = reinterpret_cast<uintptr_t>(C.data_ptr<float>());
        if ((pa & 0xf) == 0 && (pb & 0xf) == 0 && (pc & 0xf) == 0) {
            vector_add_kernel_float4<<<blocks, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
        } else {
            vector_add_kernel_generic<float><<<blocks, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
        }
    } else {
        vector_add_kernel_generic<double><<<blocks, threads>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), N);
    }

    return C;
}
"""

cpp_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
torch::Tensor vector_add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

module = load_inline(
    name="vector_add",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_add_cuda"],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--extra-device-vectorization",
    ]
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    return module.vector_add_cuda(A, B, output)

