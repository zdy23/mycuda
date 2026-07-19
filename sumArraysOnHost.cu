#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    int nElem = 1024;
    printf("Vector size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // === 1. 分配内存 ===
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C_gpu = (float *)malloc(nBytes);  // 存 GPU 结果
    h_C_cpu = (float *)malloc(nBytes);  // 存 CPU 结果

    // === 2. 初始化数据（先初始化！）===
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // === 3. GPU 内存分配 ===
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // === 4. 拷贝 CPU -> GPU ===
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // === 5. GPU 计算 ===
    int blockSize = 256;
    int gridSize = (int)ceil((float)nElem / blockSize);
    printf("Launching kernel: <<<gridSize=%d, blockSize=%d>>>\n", gridSize, blockSize);

    sumArraysOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();

    // === 6. 拷贝 GPU -> CPU ===
    cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost);

    // === 7. CPU 计算（对比验证）===
    sumArraysOnHost(h_A, h_B, h_C_cpu, nElem);

    // === 8. 结果对比 ===
    float maxError = 0.0f;
    for (int i = 0; i < nElem; i++)
    {
        float diff = fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > maxError) maxError = diff;
    }
    printf("Max error between GPU and CPU: %e\n", maxError);

    // === 9. 释放资源 ===
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}