#pragma once
#include "tensorflow/core/user_ops/get_diag_func.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void GetDiagKernel(const T* in, T* out, const int M) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < M) { // check inside diag
		int mat_idx = idx * (M + 1);
		out[idx] = in[mat_idx];
	}
}

// partial specialisation for GPU
namespace tensorflow {
namespace functors {	
	template<typename T>
	struct get_diag<GPUDevice, T> {
		void operator()(const GPUDevice& d, const T* in, T* out, const int M) {
			unsigned int threads_per_block = 256;
			unsigned int num_blocks = M / threads_per_block + 1;
			GetDiagKernel<<<num_blocks, threads_per_block, 0, d.stream()>>>(in, out, M);
		}
	};
}
}