/* Copyright 2016 Tom Nickson. All rights reserved. */
#pragma once
// #include <cudart.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/user_ops/triangle.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
template<typename T>
__global__ void TriangleKernelLower(const T * src, T * dst, const int N) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((c < N) && (r < N)) { // check in matrix bounds
    	int idx = r * N + c; // row major!
    	if (r >= c) { // on or below diagonal
    		dst[idx] = src[idx];
    	} else {
    		dst[idx] = 0;
    	}
    }
}

template<typename T>
__global__ void TriangleKernelUpper(const T * src, T * dst, const int N) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((c < N) && (r < N)) { // check in matrix bounds
    	int idx = r * N + c; // row major!
    	if (r <= c) { // check on or above diagonal
    		dst[idx] = src[idx];
    	} else {
    		dst[idx] = 0;
    	}
    }
}

namespace functors {
	template<typename T>
	struct upper_tri<GPUDevice, T> {
		void operator()(const GPUDevice& d, const T * in, T * out, int N) {
			const unsigned int threads_per_side = 16;
			const dim3 threads_per_block{threads_per_side, threads_per_side};
			dim3 num_blocks{N / threads_per_side + 1, N / threads_per_side + 1};
			TriangleKernelUpper<<<num_blocks, threads_per_block, 0, d.stream()>>>(in, out, N);

		}
	};

	template<typename T>
	struct lower_tri<GPUDevice, T> {
		void operator()(const GPUDevice& d, const T * in, T * out, int N) {
			const unsigned int threads_per_side = 16;
			const dim3 threads_per_block{threads_per_side, threads_per_side};
			dim3 num_blocks{N / threads_per_side + 1, N / threads_per_side + 1};
			TriangleKernelLower<<<num_blocks, threads_per_block, 0, d.stream()>>>(in, out, N);
		}
	};
}
}
#endif