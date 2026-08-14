#include <cuda_runtime.h>

static constexpr int RS = 16; // row size
static constexpr int CS = 1;  // col size

__global__ void _2d_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
	int input_stride,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
) {
	// thread row and col idx
    int col = (blockDim.x * blockIdx.x + threadIdx.x) * CS;
	int row = (blockDim.y * blockIdx.y + threadIdx.y) * RS;
	if (row >= input_rows || col >= input_cols) return;
	
	// use cache as slide window
	float sum[RS][CS] = {};
	float cache[RS][CS];

	// output size
	int output_rows = input_rows - kernel_rows + 1;
	int output_cols = input_cols - kernel_cols + 1;
	
	for (int c = 0; c < kernel_cols; ++c) {
		// warmup
		for (int i = 0; i < RS - 1; ++i) {
			for (int j = 0; j < CS; ++j) {
				cache[i][j] =
					input[(row + i) * input_stride + (col + c + j)];
			}
		}

		// slide by kernel
		for (int r = 0; r < kernel_rows; ++r) {

			float k = kernel[r * kernel_cols + c];

			// last row
			for (int j = 0; j < CS; ++j) {
				cache[RS - 1][j] =
					input[(row + r + (RS - 1)) * input_stride + (col + c + j)];
			}

			// mul + add
			for (int i = 0; i < RS; ++i) {
				for (int j = 0; j < CS; ++j) {
					sum[i][j] += cache[i][j] * k;
				}
			}

			// slide one
			for (int i = 0 ; i < RS - 1; ++i) {
				for (int j = 0; j < CS; ++j) {
					cache[i][j] = cache[i + 1][j];
				}
			}
		}
	}

	// write back
	for (int i = 0; i < RS; ++i) {
		for (int j = 0; j < CS; ++j) {
			int oy = row + i;
			int ox = col + CS + j;
			if (oy < output_rows && ox < output_cols) {
				output[oy * output_cols + ox] = sum[i][j];
			}
		}
	}
}


extern "C" void solve(
    const float* input, const float* kernel, float* output, int input_rows,
    int input_cols, int kernel_rows, int kernel_cols
) {
	constexpr int BX = 32;
	constexpr int BY = 8;

    int out_h = input_rows - kernel_rows + 1;
    int out_w = input_cols - kernel_cols + 1;

	dim3 block(BX, BY);
	dim3 grid(
		(out_w + BX * CS - 1) / (BX * CS),
		(out_h + BY * RS - 1) / (BY * RS)
	);

	_2d_conv_kernel<<<grid, block>>>(
		input, kernel, output,
		input_cols,
		input_rows, input_cols,
		kernel_rows, kernel_cols
	);
}
