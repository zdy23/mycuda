#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cmath>

extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta);

static float timed(const half* A, const half* B, half* C,
                   int M, int N, int K, float alpha, float beta,
                   int iters = 50) {                 // 不分配,不拷贝,只测纯 kernel
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    solve(A, B, C, M, N, K, alpha, beta);           // warmup 1
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i)
        solve(A, B, C, M, N, K, alpha, beta);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;                                // 平均单次
}

// cuBLAS 参考(可选,需要链接 -lcublas)
static float cublas_ref(cublasHandle_t h, const half* A, const half* B, half* C,
                        int M, int N, int K, int iters = 50) {
    half alpha_h = __float2half(1.f);
    half beta_h  = __float2half(0.f);
    // cuBLAS 是列主序,所以交换 A↔B 并把 M,N 看成转置后参数
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha_h,
                 B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K,
                 &beta_h,
                 C, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha_h, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                     &beta_h, C, CUDA_R_16F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

    struct S { int M, N, K; };
    std::vector<S> sizes = {
        {128,128,128}, {256,256,256}, {512,512,512},
        {1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096},
        {256,512,256}, {512,256,1024},          // 一些 skinny / 长矩形
    };

    float ALPHA = 1.f, BETA = 0.f;               // beta=0 时 cuBLAS 可走 default 算法
    for (auto s : sizes) {
        int M = s.M, N = s.N, K = s.K;
        std::vector<half> A(M*K, __float2half(1.f)), B(K*N, __float2half(1.f)), C(M*N);
        half *dA, *dB, *dC;
        cudaMalloc(&dA, M*K*sizeof(half));
        cudaMalloc(&dB, K*N*sizeof(half));
        cudaMalloc(&dC, M*N*sizeof(half));
        cudaMemcpy(dA, A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, M*N*sizeof(half));
        cudaDeviceSynchronize();

        float t_my    = timed(dA, dB, dC, M, N, K, ALPHA, BETA);
        float t_cub   = cublas_ref(h, dA, dB, dC, M, N, K);

        double gflop   = 2.0 * M * N * K;                       // 2× 都是乘加
        double my_tf   = gflop / (t_my  * 1e9);                  // ms→s: ×1e-3 → 1e-3 / 1e9 = 1e-12
        double cub_tf  = gflop / (t_cub * 1e9);

        printf("M=%4d N=%4d K=%4d | mine=%7.2f ms  %6.2f TF | cuBLAS=%6.2f ms  %6.2f TF | ratio=%.2fx\n",
               M, N, K, t_my, my_tf, t_cub, cub_tf, t_cub / t_my);
        cudaDeviceSynchronize();

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    cublasDestroy(h);
    return 0;
}