#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>


const int maxn = 100, minn = -100;
void initData(int* nums, int size) {
	srand(time(NULL));	
	for (int i = 0; i < size; i++) {
		nums[i] = rand() % (maxn - minn + 1) + minn;
	}
}

__global__ void reduceDivergent(int* input, int* output, int size) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;

	if (idx >= size) return;

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if (tid % (2 * stride) == 0) {
			input[idx] += input[idx + stride];
		}
		__syncthreads();
	}
}

__global__ void reduceNotDivergent(int* input, int* output, int size) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;

	if (idx >= size) return;

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			output[index] = input[index] + input[index + stride];
		}
		__syncthreads();
	}
}	

__global__ void reduceInterLeaved(int* input, int* output, int size) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;

	if (idx >= size) return;

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 2) {
		if (tid < stride) {
			input[idx] += input[idx + stride];
		}
		__syncthreads();
	}
}

int main() {
	int N = 256;
	int *h_input = new int[N];
	int *h_output = new int[N];
	initData(h_input, N);

	int *d_input = nullptr;
	int *d_output = nullptr;
	cudaMalloc(&d_input, N * sizeof(int));
	cudaMalloc(&d_output, N * sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	reduceDivergent<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
	cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time: %f ms\n", milliseconds);

	cudaEventRecord(start);
	reduceNotDivergent<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
	cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time: %f ms\n", milliseconds);

	cudaEventRecord(start);
	reduceInterLeaved<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
	cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time: %f ms\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_output);
	delete[] h_input;
	delete[] h_output;
}