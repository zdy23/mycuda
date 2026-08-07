#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

extern "C" void solve(const float* input, float* output, int N);

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    srand(time(nullptr));
    for (int i = 0; i < N; i++)
        h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("softmax N=%d: %.3f ms\n", N, ms);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
