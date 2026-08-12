#include <cuda_runtime.h>
#include <stdint.h>

// float4 needs 16-byte alignment
#define IS_F4_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 15u) == 0u)

// tile: fat X (full warp), thin Y/Z — good for cols≈128
constexpr int TX = 32;
constexpr int TY = 4;
constexpr int TZ = 2;
constexpr int OUT_X = TX * 4; // 128

// X dim rounded up to multiple of 4 for float4; Y/Z exact (tile + halo)
__host__ __device__ __forceinline__ int smem_x_dim(int out_x, int kx) {
	return (out_x + kx - 1 + 3) & ~3;
}

__device__ __forceinline__ float4 load_float4_safe(const float* p) {
	if (IS_F4_ALIGNED(p)) {
		return *reinterpret_cast<const float4*>(p);
	}
	return make_float4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__ void store_float4_safe(float* p, float4 v) {
	if (IS_F4_ALIGNED(p)) {
		*reinterpret_cast<float4*>(p) = v;
	} else {
		p[0] = v.x;
		p[1] = v.y;
		p[2] = v.z;
		p[3] = v.w;
	}
}

__constant__ float c_kernel[1000]; // assumes KX*KY*KZ <= 1000

__global__ void _3d_conv_kernel(
	const float* __restrict__ input,
	float* __restrict__ output,
	int input_depth, int input_rows, int input_cols,
	int output_depth, int output_rows, int output_cols,
	int KX, int KY, int KZ,
	int SMEM_X, int SMEM_Y, int SMEM_Z
) {
	// flat smem: [z][y][x] -> [z * SMEM_Y * SMEM_X + y * SMEM_X + x]
	extern __shared__ __align__(16) float smem[];

	const int out_x0 = blockIdx.x * OUT_X;
	const int out_y0 = blockIdx.y * TY;
	const int out_z0 = blockIdx.z * TZ;

	const int smem_z_stride = SMEM_Y * SMEM_X;
	const int kernel_z_stride = KY * KX;

	// cooperative load: threads map along x (warp-coalesced), then y, z
	for (int sz = threadIdx.z; sz < SMEM_Z; sz += TZ) {
		const int gz = out_z0 + sz;
		for (int sy = threadIdx.y; sy < SMEM_Y; sy += TY) {
			const int gy = out_y0 + sy;
			for (int sx = threadIdx.x * 4; sx < SMEM_X; sx += OUT_X) {
				const int gx = out_x0 + sx;
				const int smem_base = sz * smem_z_stride + sy * SMEM_X + sx;

				if (gz < input_depth && gy < input_rows && gx + 3 < input_cols) {
					const int base = (gz * input_rows + gy) * input_cols + gx;
					*reinterpret_cast<float4*>(&smem[smem_base]) =
						load_float4_safe(input + base);
				} else if (gz < input_depth && gy < input_rows && gx < input_cols) {
					const int base = (gz * input_rows + gy) * input_cols + gx;
					float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
					v.x = input[base];
					if (gx + 1 < input_cols) v.y = input[base + 1];
					if (gx + 2 < input_cols) v.z = input[base + 2];
					if (gx + 3 < input_cols) v.w = input[base + 3];
					*reinterpret_cast<float4*>(&smem[smem_base]) = v;
				} else {
					*reinterpret_cast<float4*>(&smem[smem_base]) =
						make_float4(0.f, 0.f, 0.f, 0.f);
				}
			}
		}
	}
	__syncthreads();

	// each thread computes 4 outputs along x
	const int tx4 = threadIdx.x * 4;
	const int ty = threadIdx.y;
	const int tz = threadIdx.z;

	const int out_x = out_x0 + tx4;
	const int out_y = out_y0 + ty;
	const int out_z = out_z0 + tz;

	float acc0 = 0.f;
	float acc1 = 0.f;
	float acc2 = 0.f;
	float acc3 = 0.f;

	for (int kz = 0; kz < KZ; ++kz) {
		const int smem_z = (tz + kz) * smem_z_stride;
		const int kernel_z = kz * kernel_z_stride;
		for (int ky = 0; ky < KY; ++ky) {
			const int smem_y = (ty + ky) * SMEM_X;
			const int kernel_y = ky * KX;
			for (int kx = 0; kx < KX; ++kx) {
				const float w = c_kernel[kernel_z + kernel_y + kx];
				const int sidx = smem_z + smem_y + tx4 + kx;
				acc0 = fmaf(smem[sidx + 0], w, acc0);
				acc1 = fmaf(smem[sidx + 1], w, acc1);
				acc2 = fmaf(smem[sidx + 2], w, acc2);
				acc3 = fmaf(smem[sidx + 3], w, acc3);
			}
		}
	}

	// write back
	if (out_z < output_depth && out_y < output_rows && out_x < output_cols) {
		const int base = (out_z * output_rows + out_y) * output_cols + out_x;
		if (out_x + 3 < output_cols) {
			store_float4_safe(output + base, make_float4(acc0, acc1, acc2, acc3));
		} else {
			output[base] = acc0;
			if (out_x + 1 < output_cols) output[base + 1] = acc1;
			if (out_x + 2 < output_cols) output[base + 2] = acc2;
			if (out_x + 3 < output_cols) output[base + 3] = acc3;
		}
	}
}

extern "C" void solve(
	const float* input, const float* kernel, float* output, int input_depth,
	int input_rows, int input_cols, int kernel_depth, int kernel_rows,
	int kernel_cols
) {
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_rows  = input_rows  - kernel_rows  + 1;
	const int output_cols  = input_cols  - kernel_cols  + 1;

	const dim3 block(TX, TY, TZ);
	const dim3 grid(
		(output_cols  + OUT_X - 1) / OUT_X,
		(output_rows  + TY    - 1) / TY,
		(output_depth + TZ    - 1) / TZ
	);

	const int SMEM_X = smem_x_dim(OUT_X, kernel_cols);
	const int SMEM_Y = TY + kernel_rows - 1;
	const int SMEM_Z = TZ + kernel_depth - 1;
	const size_t smem_bytes = sizeof(float) * SMEM_X * SMEM_Y * SMEM_Z;

	cudaMemcpyToSymbol(c_kernel, kernel,
		sizeof(float) * kernel_cols * kernel_rows * kernel_depth);
	_3d_conv_kernel<<<grid, block, smem_bytes>>>(
		input, output,
		input_depth, input_rows, input_cols,
		output_depth, output_rows, output_cols,
		kernel_cols, kernel_rows, kernel_depth,
		SMEM_X, SMEM_Y, SMEM_Z
	);

	cudaDeviceSynchronize();
}
