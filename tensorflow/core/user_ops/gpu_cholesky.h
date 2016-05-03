#pragma once

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
	namespace functors {
 	template <typename Device, typename T>
    struct chol_functor {
      void operator()(const Device& d, const T* in, const int M, T* out, bool& success);
    };
} // functors
} // tensorflow