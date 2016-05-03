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

typedef Eigen::ThreadPoolDevice CPUDevice;
namespace tensorflow {
// partial specialisation for CPU
namespace functors {	
	template<typename T>
	struct upper_tri<CPUDevice, T> {
		using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	    using ConstMatrixMap = Eigen::Map<const Matrix>;
	    using MatrixMap = Eigen::Map<Matrix>;
		void operator()(const CPUDevice&, const T * in, T * out, int M) {
			//The next three lines are necessary to get Eigen matrix behaviour.
		    const ConstMatrixMap in_mat(in, M, M);
		    MatrixMap out_mat(out, M, M);
			out_mat = in_mat.template triangularView<Eigen::Upper>();
		}
	};

	template<typename T>
	struct lower_tri<CPUDevice, T> {
		using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	    using ConstMatrixMap = Eigen::Map<const Matrix>;
	    using MatrixMap = Eigen::Map<Matrix>;
		void operator()(const CPUDevice&, const T * in, T * out, int M) {
			//The next three lines are necessary to get Eigen matrix behaviour.
		    const ConstMatrixMap in_mat(in, M, M);
		    MatrixMap out_mat(out, M, M);
			out_mat = in_mat.template triangularView<Eigen::Lower>();
		}
	};
}
}