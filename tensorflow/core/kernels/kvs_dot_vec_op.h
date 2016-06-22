#pragma once
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {
	template <typename Device, typename T, int D>
	struct kvs_dot_vec_functor {
		void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
			const Tensor* vec, Tensor* product);
	};

	template <typename Device, typename T, int D>
	struct kvs_dot_vec_kvsgrad_functor {
		void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
			const Tensor* vec, const Tensor* ingrad, OpOutputList* kvsgrad);
	};

	template <typename Device, typename T, int D>
	struct kvs_dot_vec_vecgrad_functor {
		void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
			const Tensor* vec, Tensor* vecgrad);
	};


	template <typename T, int D>
	void launch_cu_kvs_dot_vec(int H, 
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* d_vec, T* d_out);

	template <typename T, int D>
	void launch_cu_kvs_dot_vec_kvsgrad(int H, 
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* d_vec, const T* in_grad,
		T** kvs_grad);

	template <typename T, int D>
	void launch_cu_kvs_dot_vec_vecgrad(int H, int64_t vec_len,
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* in_grad,
		T * vec_grad);
}