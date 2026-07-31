#include <cuda_fp16.h>

/*
    CUDA kernel for vector addition with optimizations for float and half types.
    - For float, it uses float4 vectorization to process 4 floats at a time.
    - For half, it uses half8 vectorization to process 8 halfs at a time.
    - It also includes a generic kernel for other types or unaligned data.
*/

// scalar_t 类型的通用向量加法 kernel
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

// float4 向量化：每线程每轮处理 2 个 float4 ILP 翻倍
__global__ __launch_bounds__(256, 8) void vector_add_kernel_float4(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int N) {
  const float4 *A4 = reinterpret_cast<const float4 *>(A);
  const float4 *B4 = reinterpret_cast<const float4 *>(B);
  float4 *C4 = reinterpret_cast<float4 *>(C);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int N4 = N / 4;

// 每次迭代处理 i 和 i+stride 两个 float4
#pragma unroll
  for (int i = idx; i < N4; i += stride * 2) {
    // 处理 i 的 float4
    float4 a0 = A4[i];
    float4 b0 = B4[i];
    float4 c0;
    c0.x = a0.x + b0.x;
    c0.y = a0.y + b0.y;
    c0.z = a0.z + b0.z;
    c0.w = a0.w + b0.w;
    C4[i] = c0;

    // 处理 i + stride 的 float4
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

  // 尾部 float 标量
  int tail = N4 * 4;
  for (int i = tail + idx; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

// half8 向量化：每线程每轮处理 2 个 half8 ILP 翻倍
__global__ __launch_bounds__(256, 8) void vector_add_kernel_half8(
    const half *__restrict__ A, const half *__restrict__ B,
    half *__restrict__ C, int N) {
  const uint4 *A8 = reinterpret_cast<const uint4 *>(A);
  const uint4 *B8 = reinterpret_cast<const uint4 *>(B);
  uint4 *C8 = reinterpret_cast<uint4 *>(C);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int N8 = N / 8;

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

  // 尾部 half2
  const half2 *A2 = reinterpret_cast<const half2 *>(A);
  const half2 *B2 = reinterpret_cast<const half2 *>(B);
  half2 *C2 = reinterpret_cast<half2 *>(C);

  int tail_half2 = N8 * 4;
  for (int i = tail_half2 + idx; i < N / 2; i += stride) {
    C2[i] = __hadd2(A2[i], B2[i]);
  }

  // 尾部 half
  int tail_half = (N / 2) * 2;
  for (int i = tail_half + idx; i < N; i += stride) {
    C[i] = __hadd(A[i], B[i]);
  }
}