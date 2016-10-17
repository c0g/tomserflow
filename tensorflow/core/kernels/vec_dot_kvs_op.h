#pragma once
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {
	template <typename Device, typename T, int D>
	struct vec_dot_kvs_functor {
		void operator()(OpKernelContext* ctx, const Tensor* vec,
			const OpInputList& kvs, Tensor* product);
	};

	template <typename Device, typename T, int D>
	struct vec_dot_kvs_kvsgrad_functor {
		void operator()(OpKernelContext* ctx, const Tensor* vec,
			const OpInputList& kvs, const Tensor* ingrad, OpOutputList* kvsgrad);
	};

	template <typename Device, typename T, int D>
	struct vec_dot_kvs_vecgrad_functor {
		void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
			const Tensor* ingrad, Tensor* vecgrad);
	};


	template <typename T, int D>
	void launch_cu_vec_dot_kvs(const Eigen::GpuDevice& dev, const int W, const int64_t vec_len,
		const int * d_heights, const int64_t * d_hprods,
		const T* d_vec, const T *const * d_kvs, T* d_out);

	template <typename T, int D>
	void launch_cu_vec_dot_kvs_kvsgrad(const Eigen::GpuDevice& dev, const int W, const int64_t vec_len,
		const int * heights, const int * d_heights, const int64_t * d_hprods,
		const T* d_vec, const T * const * d_kvs, const T* in_grad,
		T** kvs_grad);

	template <typename T, int D>
	void launch_cu_vec_dot_kvs_vecgrad(const Eigen::GpuDevice& dev, const int W, const int64_t vec_len,
		const int * d_heights, const int64_t * d_hprods,
		const T * const * d_kvs, const T* in_grad,
		T * vec_grad);
}