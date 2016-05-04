#pragma once
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
	namespace functors {
 	template <typename Device, typename T>
    struct chol_functor {
      void operator()(OpKernelContext*, const T*, const int M, T*, bool&); 
    };
} // functors
} // tensorflow