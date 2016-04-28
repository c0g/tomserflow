#pragma once
#include "tensorflow/core/user_ops/triangle_func.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::ThreadPoolDevice CPUDevice;

// partial specialisation for CPU
namespace functors {	
	template<typename T>
	struct get_diag<CPUDevice, T> {
		void operator()(const CPUDevice & /*ignored*/, const T * in, T * out, const int M) {
			for (int idx = 0; idx < M; ++idx) { // maybe inefficient but also O(N) so who cares
				int mat_idx = idx * (M + 1);
				out[idx] = in[mat_idx];
			}
		}
	};
}