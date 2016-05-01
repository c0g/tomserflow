#pragma once
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace functors {	
    template<typename Device, typename T>
	struct get_diag {
		void operator()(const Device& d, const T * in, T * out, int M);
	};
}