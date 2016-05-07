#pragma once
namespace tensorflow {
	namespace functor {
		template<typename Dev, typename T>
		struct SetDiag {
		    void compute(const Dev& d, 
		    	size_t N, const T* diag, T* tensor);
		};
		template<typename Dev, typename T>
		struct GetDiag {
		    void compute(const Dev& d, 
		    	size_t N, const T* tensor, T* diag);
		};
	} // functor
} // tensorflow