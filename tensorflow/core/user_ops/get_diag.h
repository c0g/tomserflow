#pragma once
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
namespace tensorflow {
namespace functors {	
	typedef Eigen::ThreadPoolDevice CPUDevice;
    template<typename Device, typename T>
	struct get_diag {
		void operator()(const Device& d, const T * in, T * out, int M);
	};
	template<typename T>
	struct get_diag<CPUDevice, T> {
		void operator()(const CPUDevice & /*ignored*/, const T * in, T * out, const int M) {
			for (int idx = 0; idx < M; ++idx) {
				int mat_idx = idx * (M + 1);
				out[idx] = in[mat_idx];
			}
		}
	};
} // namespace functors
} // namespace tensorflow