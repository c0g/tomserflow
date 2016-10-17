#define EIGEN_USE_GPU
#include "vec_dot_kvs_op.h"
#include "tensorflow/core/kernels/fill_functor.h"

/*
Calculate the product of a vector and a vertical row-major KVS.

   	             	  /  k k k  \ 
	  		  		  |  k k k  |
		  			  |		    |
v v v v    <dot>      |  <kvs>  |  =  a a a
					  |         |
					  |  k k k  |
					  \  k k k  /

Splits the KVS up into BLOCK_SIZE_PARLOADxBLOCK_SIZE_PARLOAD blocks. 
A BLOCK_SIZE_PARLOAD thread block calculates the BLOCK_SIZE_PARLOAD 
partial product for each block, by moving down the block.

To avoid the need for a large itermediate buffer to store each block's
answer it directly stores the answer into global memory using atomics.

This leads to non-deterministic answers, so don't be surprised!
*/

const int BLOCK_SIZE_PARLOAD = 512;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


// General function to iterate through kvs layers
template <typename T, int thisD>
__device__ bool cu_vec_dot_kvs_par_inner(const int D, const int W, const int64_t vec_len,
	const int col, const int * heights, const int64_t * hprods, int * starts,
	const T * vec, const T * const * kvs, T & acc, 
	const T kvs_prod, int64_t base_vec_idx) {
	for (int kvs_idx = starts[D - thisD]; kvs_idx < heights[D - thisD]; ++kvs_idx) {
		if (cu_vec_dot_kvs_par_inner<T, thisD - 1>(D, W, vec_len, col, heights, 
			hprods, starts, vec, kvs, acc, 
			kvs_prod * kvs[D-thisD][col + W * kvs_idx],
			base_vec_idx + kvs_idx * hprods[D - thisD])) {
			return true;
		}
	}
	starts[D - thisD] = 0;
	return false;
}

// Float specialisation for last KVS
template <>
__device__ bool cu_vec_dot_kvs_par_inner<float, 1>(const int D, const int W, const int64_t vec_len, 
	const int col, const int * heights, const int64_t * hprods,  int * starts,
	const float * vec, const float * const * kvs, float & acc, 
	const float kvs_prod, int64_t base_vec_idx) {
	
	for (int kvs_idx = starts[D-1]; kvs_idx < heights[D - 1]; ++kvs_idx) {
		int64_t vec_idx = base_vec_idx + kvs_idx * hprods[D - 1];
		acc += kvs[D-1][col + W * kvs_idx] * kvs_prod * vec[vec_idx];
		if ((((vec_idx + hprods[D - 1]) % BLOCK_SIZE_PARLOAD) == 0) ||
			 ((vec_idx + hprods[D - 1]) == vec_len)) {
			return true;
		}
	}
	starts[D - 1] = 0;
	return false;
}

// Double specialisation for last KVS
template <>
__device__ bool cu_vec_dot_kvs_par_inner<double, 1>(const int D, const int W, const int64_t vec_len, 
	const int col, const int * heights, const int64_t * hprods,  int * starts,
	const double * vec, const double * const * kvs, double & acc, 
	const double kvs_prod, int64_t base_vec_idx) {
	
	for (int kvs_idx = starts[D-1]; kvs_idx < heights[D - 1]; ++kvs_idx) {
		int64_t vec_idx = base_vec_idx + kvs_idx * hprods[D - 1];
		acc += kvs[D-1][col + W * kvs_idx] * kvs_prod * vec[vec_idx];
		if ((((vec_idx + hprods[D - 1]) % BLOCK_SIZE_PARLOAD) == 0) ||
			 ((vec_idx + hprods[D - 1]) == vec_len)) {
			return true;
		}
	}
	starts[D - 1] = 0;
	return false;
}

// Wrapper
template <typename T, int D>
__global__ void cu_vec_dot_kvs_par(const int W, const int64_t vec_len, 
		const int * heights, const int64_t * hprods, 
		const T * vec, const T * const * kvs, T* out) {
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t vec_start = blockIdx.x * BLOCK_SIZE_PARLOAD;
	if (col < W) {
		T acc = 0.0;
		int starts[D];
		for (int d = 0; d < D; ++d) {
			starts[d] = (vec_start / hprods[d]) % heights[d];
		}
		cu_vec_dot_kvs_par_inner<T, D>(D, W, vec_len, col, heights, hprods, starts, 
			vec, kvs, acc, T(1.0), int64_t(0));
		atomicAdd(&out[col], acc);
	}
}

/*
Propagates the gradient from the output to the kvs specified by targetD.

   	                  /  k k k  \ 
	  		  		  |  k k k  |
		  			  |		    |
v v v v    <dot>      |  <kvs>  |  =  g g g
					  |         |
					  |  k k k  |
					  \  k k k  /

Given the gradients in g, propagates these into the targeted dimension of ks.

A thread block calculates the value for a 1xBLOCK_SIZE_KVSGRAD segment of the matrix.
*/
const int BLOCK_SIZE_KVSGRAD = 512;

template <typename T, int thisD>
__device__ void cu_vec_dot_kvs_kvsgrad_inner(const int D, const int W,
	const int col, const int * heights, const int64_t * hprods,
	const T * vec, const T * const * kvs, T & acc, 
	const T kvs_prod, int64_t base_vec_idx) {

	for (int kvs_idx = 0; kvs_idx < heights[D - thisD]; ++kvs_idx) {
		cu_vec_dot_kvs_kvsgrad_inner<T, thisD - 1>(D, W, col, heights, 
			hprods, vec, kvs, acc, 
			kvs_prod * kvs[D-thisD][col + W * kvs_idx],
			base_vec_idx + kvs_idx * hprods[D - thisD]);
	}
}

template <>
__device__ void cu_vec_dot_kvs_kvsgrad_inner<float, 1>(const int D, const int W,
	const int col, const int * heights, const int64_t * hprods,
	const float * vec, const float * const * kvs, float & acc, 
	const float kvs_prod, int64_t base_vec_idx){

	for (int kvs_idx = 0; kvs_idx < heights[D - 1]; ++kvs_idx) {
		int64_t vec_idx = base_vec_idx + kvs_idx * hprods[D - 1];
		acc += kvs[D-1][col + W * kvs_idx] * kvs_prod * vec[vec_idx];
	}
}
template <>
__device__ void cu_vec_dot_kvs_kvsgrad_inner<double, 1>(const int D, const int W,
	const int col, const int * heights, const int64_t * hprods,
	const double * vec, const double * const * kvs, double & acc, 
	const double kvs_prod, int64_t base_vec_idx){

	for (int kvs_idx = 0; kvs_idx < heights[D - 1]; ++kvs_idx) {
		int64_t vec_idx = base_vec_idx + kvs_idx * hprods[D - 1];
		acc += kvs[D-1][col + W * kvs_idx] * kvs_prod * vec[vec_idx];
	}
}

template <typename T, int D>
__global__ void cu_vec_dot_kvs_kvsgrad(int targetD, int W,  const int * hcols,  const int64_t * hprods,
		const T* vec, const T * const * kvs, const T* grad,
		T ** gkvs) {
	int row = blockIdx.x; // which row of kvs are we targetting
	int col = blockIdx.y * blockDim.y + threadIdx.y; // which column of kvs/grad
	const T * kvs_other[D-1];
	int hcols_other[D-1];
	int64_t hprods_other[D-1];
	if (col < W) {
		int next_d = 0;
		for (int d = 0; d < D; ++d) {
			if (d != targetD) {
				kvs_other[next_d] = kvs[d];
				hcols_other[next_d] = hcols[d];
				hprods_other[next_d] = hprods[d];
				++next_d;
			}
		}
		T acc = 0.0f;
		cu_vec_dot_kvs_kvsgrad_inner<T, D - 1>(D - 1, W, col, hcols_other, 
			hprods_other, vec, kvs_other, acc, grad[col], row * hprods[targetD]);
		gkvs[targetD][col + W * row] = acc;
	}
}

/*
Propagates the gradient g from the output to the elements of v

   	                  /  k k k  \ 
	  		  		  |  k k k  |
		  			  |		    |
v v v v    <dot>      |  <kvs>  |  =  g g g
					  |         |
					  |  k k k  |
					  \  k k k  /

With a vector N x 1 and kvs N x M, launch N threadblocks of size BLOCK_SIZE_VECGRAD
kvs * grad products are summed into shared memory.
If BLOCK_SIZE_VECGRAD is narrower than the KVS, it blockwise moves right to sum the entire 
KVS set to the shared memory. 
The shared memory is reduced in parallel to a single value which is stored in
global memory by thread0 of the block.
*/

const int BLOCK_SIZE_VECGRAD = 128;

template <typename T, int D>
__global__ void cu_vec_dot_kvs_vecgrad(const int W,  const int64_t vec_len, 
		const int * hcols,  const int64_t * hprods,
		const T * const * kvs, const T* grad,
		T * vecgrad) {
	int64_t n = blockIdx.x;
	if (n < vec_len) {

		int col = threadIdx.x;
		// printf("%lld, %d \n", n, col);
		__shared__ T accum[BLOCK_SIZE_VECGRAD];
		int kvs_offsets[D];
		for (int d = 0; d < D; ++d) {
			kvs_offsets[d] = (n / hprods[d]) % hcols[d];
		}
		accum[col] = 0;
		for (int offset = 0; offset < W; offset += BLOCK_SIZE_VECGRAD) {
			if ((col + offset) < W) {
				// printf("%d\n",col+offset);
				T kvs_tmp = grad[col + offset];
				#pragma unroll
				for (int d = 0; d < D; ++d) {
					kvs_tmp *= kvs[d][col + offset + kvs_offsets[d] * W];
				}
				accum[col] += kvs_tmp; 
			}
		}
		

		__syncthreads();
		#pragma unroll
		for (int i = BLOCK_SIZE_VECGRAD/2; i > 1; i /= 2) {
			if (col < i) {
				accum[col] += accum[col + i]; 
			}
		}
		if (col < 1) {
			vecgrad[n] = accum[col] + accum[col+1];
		}
	}
}

namespace tensorflow {
	template <typename T, int D>
	void launch_cu_vec_dot_kvs(const Eigen::GpuDevice & dev, const int W, const int64_t vec_len,
		const int * d_heights, const int64_t * d_hprods,
		const T* d_vec, const T * const * d_kvs, T* d_out) {
		dim3 threads{1, BLOCK_SIZE_PARLOAD, 1};
		dim3 nblocks{std::uint32_t(vec_len / BLOCK_SIZE_PARLOAD + ((vec_len % BLOCK_SIZE_PARLOAD) != 0)), 
			std::uint32_t(W/BLOCK_SIZE_PARLOAD + ((W % BLOCK_SIZE_PARLOAD) != 0)), 1};
		cu_vec_dot_kvs_par<T, D><<<nblocks, threads, 0, dev.stream()>>>(W, vec_len, d_heights, d_hprods,
								d_vec, d_kvs, d_out);
	}

	template <typename T, int D>
	void launch_cu_vec_dot_kvs_kvsgrad(const Eigen::GpuDevice & dev, const int W, const int64_t vec_len,
		const int * heights, const int * d_heights, const int64_t * d_hprods,
		const T* d_vec, const T * const * d_kvs, const T* in_grad,
		T** kvs_grad) {
		for (int targetD = 0; targetD < D; ++targetD) {
			dim3 threads{1, BLOCK_SIZE_KVSGRAD, 1};
			dim3 nblocks{std::uint32_t(heights[targetD]),
				std::uint32_t(W)/BLOCK_SIZE_KVSGRAD + ((W % BLOCK_SIZE_KVSGRAD) != 0), 1};
			cu_vec_dot_kvs_kvsgrad<T, D><<<nblocks, threads, 0, dev.stream()>>>(targetD, W, d_heights, d_hprods, 
								d_vec, d_kvs, in_grad, kvs_grad);
		}
	}

	template <typename T, int D>
	void launch_cu_vec_dot_kvs_vecgrad(const Eigen::GpuDevice & dev, const int W, const int64_t vec_len,
		const int * d_heights, const int64_t * d_hprods,
		const T * const * d_kvs, const T* in_grad,
		T* vec_grad) {
		cu_vec_dot_kvs_vecgrad<T, D><<<vec_len, BLOCK_SIZE_VECGRAD, 0, dev.stream()>>>(
			W, vec_len, d_heights, d_hprods, d_kvs, in_grad, vec_grad);
	}

		namespace functor {

		template <typename T>
		struct SetZeroFunctor<Eigen::GpuDevice, T> {
		  void operator()(const Eigen::GpuDevice& d, typename TTypes<T>::Flat out) {
		    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
		  }
		};
	} // functor

}
#define DEF_VEC_DOT_KVS(T, D) 										\
template void tensorflow::launch_cu_vec_dot_kvs<T, D>(const Eigen::GpuDevice & dev,\
		const int W,  \
		const int64_t vec_len,										\
		const int * d_heights, const int64_t * d_hprods,			\
		const T* d_vec, const T * const * d_kvs, T* d_out);			\
template void tensorflow::launch_cu_vec_dot_kvs_kvsgrad<T, D>(const Eigen::GpuDevice & dev,\
		const int W, const int64_t vec_len, 						\
		const int * heights,										\
		const int * d_heights, const int64_t * d_hprods,			\
		const T* d_vec, const T * const * d_kvs, const T* in_grad,	\
		T** kvs_grad);												\
template void tensorflow::launch_cu_vec_dot_kvs_vecgrad<T, D>(const Eigen::GpuDevice & dev,\
				\
		const int W, const int64_t vec_len, 						\
		const int * d_heights, const int64_t * d_hprods,			\
		const T * const * d_kvs, const T* in_grad,					\
		T* vec_grad);

DEF_VEC_DOT_KVS(float, 2)
DEF_VEC_DOT_KVS(float, 3)
DEF_VEC_DOT_KVS(float, 4)
DEF_VEC_DOT_KVS(float, 5)
DEF_VEC_DOT_KVS(float, 6)
DEF_VEC_DOT_KVS(float, 7)
DEF_VEC_DOT_KVS(float, 8)
DEF_VEC_DOT_KVS(float, 9)
DEF_VEC_DOT_KVS(float, 10)
DEF_VEC_DOT_KVS(double, 2)
DEF_VEC_DOT_KVS(double, 3)
DEF_VEC_DOT_KVS(double, 4)
DEF_VEC_DOT_KVS(double, 5)
DEF_VEC_DOT_KVS(double, 6)
DEF_VEC_DOT_KVS(double, 7)
DEF_VEC_DOT_KVS(double, 8)
DEF_VEC_DOT_KVS(double, 9)
DEF_VEC_DOT_KVS(double, 10)


#undef DEF_VEC_DOT_KVS

template struct tensorflow::functor::SetZeroFunctor<Eigen::GpuDevice, float>;	
template struct tensorflow::functor::SetZeroFunctor<Eigen::GpuDevice, double>;