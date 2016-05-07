#pragma once
#include "tensorflow/core/framework/types.h"
namespace tensorflow {
	namespace functor {
		template<typename Dev, typename T>
		struct SetDiag {
		    void operator()(const Dev& d, 
		    	uint64 N, const T* diag, T* tensor);
		};
		template<typename Dev, typename T>
		struct GetDiag {
		    void operator()(const Dev& d, 
		    	uint64 N, const T* tensor, T* diag);
		};
	} // functor
} // tensorflow
