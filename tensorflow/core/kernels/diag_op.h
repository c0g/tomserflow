// #pragma once
// namespace tensorflow {
// 	namespace functor {
// 		template<typename Dev, typename T>
// 		struct SetDiag {
// 		    void operator()(const Dev& d, 
// 		    	size_t N, size_t prank, const T* diag, T* tensor);
// 		};
// 		template<typename Dev, typename T>
// 		struct GetDiag {
// 		    void operator()(const Dev& d, 
// 		    	size_t N, size_t prank, const T* tensor, T* diag);
// 		};
// 	} // functor
// } // tensorflow