#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// solve.cu 里的 extern "C" solve，带模板配置参数
extern "C" void solve(
	const float* Q, const float* K, const float* V,
	float* output, int M, int N, int d,
	int tile_size, int tiles_per_thread, int threads_per_block);

static void usage(const char* prog) {
	fprintf(stderr,
		"usage: %s M N d tile_size tiles_per_thread threads_per_block [verify] [warmup] [repeat]\n"
		"  M, N, d            矩阵维度 (Q: M * d, K/V: N * d, O: M * d)\n"
		"  tile_size          GEMM tile 边长 (16/32)\n"
		"  tiles_per_thread   线程粗化因子 (1/2/4)\n"
		"  threads_per_block  softmax block 线程数 (128/256/512)\n"
		"  verify             0/1 是否与 CPU 参考对比 (默认 1)\n"
		"  warmup             预热次数 (默认 3)\n"
		"  repeat             计时重复次数 (默认 10)\n"
		"输出: 一行 JSON, 便于 python 解析\n", prog);
}

// CPU 参考实现: O = softmax(Q K^T / sqrt(d)) V
static void cpu_ref(const std::vector<float>& Q, const std::vector<float>& K,
					const std::vector<float>& V, std::vector<float>& O,
					int M, int N, int d) {
	float scale = sqrtf((float)d);
	std::vector<float> score(N);
	for (int i = 0; i < M; i++) {
		float m = -1e30f;
		for (int j = 0; j < N; j++) {
			float s = 0.0f;
			for (int k = 0; k < d; k++)
				s += Q[i * d + k] * K[j * d + k];
			s /= scale;
			score[j] = s;
			if (s > m) m = s;
		}
		float l = 0.0f;
		for (int j = 0; j < N; j++) { score[j] = expf(score[j] - m); l += score[j]; }
		for (int j = 0; j < N; j++) score[j] /= l;
		for (int k = 0; k < d; k++) {
			float o = 0.0f;
			for (int j = 0; j < N; j++) o += score[j] * V[j * d + k];
			O[i * d + k] = o;
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 7) { usage(argv[0]); return 1; }

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int d = atoi(argv[3]);
	int tile_size = atoi(argv[4]);
	int tpt = atoi(argv[5]);
	int tpb = atoi(argv[6]);
	int verify = argc > 7 ? atoi(argv[7]) : 1;
	int warmup = argc > 8 ? atoi(argv[8]) : 3;
	int repeat = argc > 9 ? atoi(argv[9]) : 10;

	if (M <= 0 || N <= 0 || d <= 0) { fprintf(stderr, "bad dims\n"); return 1; }

	size_t qk_bytes = (size_t)M * d * sizeof(float);
	size_t v_bytes = (size_t)N * d * sizeof(float);
	size_t o_bytes = qk_bytes;

	std::vector<float> h_Q((size_t)M * d), h_K((size_t)N * d), h_V((size_t)N * d);
	std::vector<float> h_O((size_t)M * d);

	srand(42);
	auto randf = []() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
	for (auto& x : h_Q) x = randf();
	for (auto& x : h_K) x = randf();
	for (auto& x : h_V) x = randf();

	float *d_Q, *d_K, *d_V, *d_O;
	cudaMalloc(&d_Q, qk_bytes);
	cudaMalloc(&d_K, v_bytes);
	cudaMalloc(&d_V, v_bytes);
	cudaMalloc(&d_O, o_bytes);
	cudaMemcpy(d_Q, h_Q.data(), qk_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_K, h_K.data(), v_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V.data(), v_bytes, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// warmup
	for (int i = 0; i < warmup; i++)
		solve(d_Q, d_K, d_V, d_O, M, N, d, tile_size, tpt, tpb);
	cudaDeviceSynchronize();

	// timed
	float best = 1e30f;
	for (int i = 0; i < repeat; i++) {
		cudaEventRecord(start);
		solve(d_Q, d_K, d_V, d_O, M, N, d, tile_size, tpt, tpb);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		if (ms < best) best = ms;
	}

	// correctness
	double max_err = -1.0;
	if (verify) {
		cudaMemcpy(h_O.data(), d_O, o_bytes, cudaMemcpyDeviceToHost);
		std::vector<float> ref((size_t)M * d);
		cpu_ref(h_Q, h_K, h_V, ref, M, N, d);
		for (size_t i = 0; i < h_O.size(); i++) {
			double e = fabs((double)h_O[i] - ref[i]);
			if (e > max_err) max_err = e;
		}
	}

	// JSON 输出
	printf("{\"M\":%d,\"N\":%d,\"d\":%d,\"ts\":%d,\"tpt\":%d,\"tpb\":%d,"
		   "\"ms\":%.4f,\"err\":%s}\n",
		M, N, d, tile_size, tpt, tpb, best,
		verify ? (max_err < 1e-3 ? "\"ok\"" : "\"FAIL\"") : "\"skip\"");

	if (verify && max_err >= 1e-3) {
		fprintf(stderr, "VERIFY FAILED: max_err=%e\n", max_err);
	}

	cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	return 0;
}