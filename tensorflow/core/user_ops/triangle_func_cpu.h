#pragma once
#include "tensorflow/core/user_ops/triangle_func.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
typedef Eigen::ThreadPoolDevice CPUDevice;

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