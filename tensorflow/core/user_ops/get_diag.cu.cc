/* Copyright 2016 Tom Nickson. All rights reserved. */

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

template <typename T>
__global__ void GetDiagKernel(const T * in, const int N, const int stride, T * out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int mat_idx = idx * (stride+1);
	if (idx < N) {
		out[idx] = in[mat_idx];
	}
}

template <typename T>
void GetDiagKernelLauncher(const T * in, const int N, const int stride, T * out) {
	const size_t threads_per_block = 256;
	size_t num_blocks = N / threads_per_block + 1;
	GetDiagKernel<<<num_blocks, threads_per_block>>>(in, N, stride, out);
}

#endif