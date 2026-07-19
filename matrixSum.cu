#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                          \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(1);                                                               \
    }                                                                          \
}

// checkresult函数用于验证GPU计算结果是否正确
void checkResult(float *hostRef, float *gpuRef, int nxy) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < nxy; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

// 二维矩阵求和核函数
__global__ void sumMatrix2D(float *A, float *B, float *C, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;
    
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

// 初始化矩阵
void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// 主机端矩阵求和（用于验证结果）
void sumMatrixOnHost(float *A, float *B, float *C, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    
    // 设置设备
    int dev = 0;
    cudaSetDevice(dev);
    
    // 矩阵维度
    int nx = 1 << 12;   // 4096 列
    int ny = 1 << 12;   // 4096 行
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    
    printf("Matrix size: nx %d ny %d\n", nx, ny);
    
    // 主机内存分配
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);
    
    // 初始化数据
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);
    
    // 设备内存分配
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));
    
    // 数据拷贝到设备
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    
    // 执行配置：二维线程块和网格
    dim3 block(32, 32);  // 1024 线程/block（达到上限）
    dim3 grid((nx + block.x - 1) / block.x, 
              (ny + block.y - 1) / block.y);
    
    printf("Execution configuration <<<(%d, %d), (%d, %d)>>>\n",
           grid.x, grid.y, block.x, block.y);
    
    // 启动核函数
    sumMatrix2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    
    // 主机端计算参考结果
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    

    // 检查结果
    checkResult(hostRef, gpuRef, nxy);
    
    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    cudaDeviceReset();
    return 0;
}
