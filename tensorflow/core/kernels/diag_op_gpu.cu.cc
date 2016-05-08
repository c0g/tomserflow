/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc
// 
#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/diag_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"

namespace tensorflow {
	namespace {
		// sets N items from diag to the diagonal of target.
		// N is product of input ranks
		// prank is product of output ranks
		// Index of diag is thread_idx, 
		// index of target is: thread_idx * (prod(ranks) + 1)
		template <typename T>
		__global__ void cu_set_diag(const uint64 N, 
										const T* diag, T* tensor) {
			uint64 idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < N) {
				uint64 didx = idx;
				uint64 tidx = idx * (N + 1);
				tensor[tidx] = diag[didx];
			}
		}
		// sets N items from the diagonal of source tensor to diagonal.
		// N is product of diagonal tensor ranks
		// prank is product of the source tensor ranks
		// Index of source is: thread_idx * (prod(ranks) + 1)
		// index of target is: thread_idx
		template <typename T>
		__global__ void cu_get_diag(const uint64 N, 
										const T* tensor, T* diag) {
			uint64 idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < N) {
				uint64 didx = idx;
				uint64 tidx = idx * (N + 1);
				diag[didx] = tensor[tidx];
			}
		}
	}// namespace
	namespace functor {

		template <typename T>
		struct SetZeroFunctor<Eigen::GpuDevice, T> {
		  void operator()(const Eigen::GpuDevice& d, typename TTypes<T>::Flat out) {
		    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
		  }
		};
		template<typename T>
		struct SetDiag<Eigen::GpuDevice, T>{
		    void operator()(const Eigen::GpuDevice& d, 
		    	uint64 N, const T* diag, T* tensor) {
		    	const uint64 tpb = 256;
		    	const uint64 nblock = tpb / N + 1;
		    	cu_set_diag<<<nblock, tpb, 0, d.stream()>>>(N, diag, tensor);
		    }
		};
		template<typename T>
		struct GetDiag<Eigen::GpuDevice, T>{
		    void operator()(const Eigen::GpuDevice& d, 
		    	uint64 N, const T* tensor, T* diag)  {
		    	const uint64 tpb = 256;
		    	const uint64 nblock = tpb / N + 1;
		    	cu_get_diag<<<nblock, tpb, 0, d.stream()>>>(N, tensor, diag);
		    }
		};
	} // functor
template struct functor::GetDiag<Eigen::GpuDevice, float>;							
template struct functor::SetDiag<Eigen::GpuDevice, float>;
template struct functor::GetDiag<Eigen::GpuDevice, double>;								
template struct functor::SetDiag<Eigen::GpuDevice, double>;
template struct functor::GetDiag<Eigen::GpuDevice, int32>;								
template struct functor::SetDiag<Eigen::GpuDevice, int32>;
template struct functor::GetDiag<Eigen::GpuDevice, int64>;								
template struct functor::SetDiag<Eigen::GpuDevice, int64>;
template struct functor::SetZeroFunctor<Eigen::GpuDevice, float>;	
template struct functor::SetZeroFunctor<Eigen::GpuDevice, double>;	
template struct functor::SetZeroFunctor<Eigen::GpuDevice, int32>;	
template struct functor::SetZeroFunctor<Eigen::GpuDevice, int64>;	
}//tensorflow 
