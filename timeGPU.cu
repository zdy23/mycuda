#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

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

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main() {
    
    int nElem = 1 << 24;
    
    float *h_A, *h_B, *hostRef, *gpuRef;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(nElem * sizeof(float));
    h_B = (float *)malloc(nElem * sizeof(float));
    hostRef = (float *)malloc(nElem * sizeof(float));
    gpuRef = (float *)malloc(nElem * sizeof(float));

    cudaMalloc((float**)&d_A, nElem * sizeof(float));
    cudaMalloc((float**)&d_B, nElem * sizeof(float));
    cudaMalloc((float**)&d_C, nElem * sizeof(float));

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    
    cudaMemcpy(d_A, h_A, nElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nElem * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (blockSize + nElem - 1) / blockSize;

    double iStart = cpuSecond();
    sumArraysOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    double iElaps = cpuSecond() - iStart;

    cudaMemcpy(gpuRef, d_C, nElem * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Time elapsed: %f sec\n", iElaps);
    printf("Performance: %f GFLOPS\n", 1.0e-9 * nElem / iElaps);
    printf("gpuRef: %f\n", gpuRef[0]);

    return 0;
}

