#pragma once
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
namespace tensorflow {
namespace functors {	
    template<typename Device, typename T>
	struct upper_tri {
		void operator()(const Device& d, const T * in, T * out, int M);
	};

	template<typename Device, typename T>
	struct lower_tri {
		void operator()(const Device& d, const T * in, T * out, int M);
	};
}
}