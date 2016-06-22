#define EIGEN_USE_GPU
#include "kvs_dot_vec_op.h"

template <typename T, int thisD>
__device__  __forceinline__ void cu_kvs_dot_vec_inner(int D, int H, int idx, 
								const int * wcols, const int64_t * wprods,
								T** kvs, const T* vec, T* out_tmp, 
								T kvs_prod, int_fast64_t vec_base) {
	for (uint_fast16_t kvs_idx = 0; kvs_idx < wcols[D-thisD]; ++kvs_idx){
		cu_kvs_dot_vec_inner<T, thisD - 1>(D, H, idx, wcols, 
				wprods, kvs, vec, out_tmp,
				kvs_prod * kvs[D-thisD][idx + H * kvs_idx], 
				vec_base + wprods[D-thisD] * kvs_idx);
	}
}

// Specialisation for last KVS
template <>
__device__  __forceinline__ void cu_kvs_dot_vec_inner<float, 1>(int D, int H, int idx, 
								const int * wcols, const int64_t * wprods,
								float** kvs, const float* vec, float* out_tmp, 
								float kvs_prod, int_fast64_t vec_base) {
	for (uint_fast16_t kvs_idx = 0; kvs_idx < wcols[D-1]; ++kvs_idx){
		*out_tmp += kvs[D-1][idx + H * (kvs_idx)] * kvs_prod * vec[vec_base++];
	}
}
template <>
__device__  __forceinline__ void cu_kvs_dot_vec_inner<double, 1>(int D, int H, int idx, 
								const int * wcols, const int64_t * wprods,
								double** kvs, const double* vec, double* out_tmp, 
								double kvs_prod, int_fast64_t vec_base) {
	for (uint_fast16_t kvs_idx = 0; kvs_idx < wcols[D-1]; ++kvs_idx){
		*out_tmp += kvs[D-1][idx + H * (kvs_idx)] * kvs_prod * vec[vec_base++];
	}
}

template <typename T, int D>
__global__ void cu_kvs_dot_vec( int H, const int * wcols, const int64_t * wprods,
	T** kvs, const T* vec, T*out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // which row of kvs/output
	if (idx < H) {
		T out_tmp = 0;
		cu_kvs_dot_vec_inner<T, D>(D, H, idx, wcols, wprods, 
			kvs, vec, &out_tmp, T(1.f), int_fast64_t(0));
		out[idx] = out_tmp;
	}
}

template <typename T, int D>
__global__ void cu_kvs_dot_vec_kvsgrad(int H,  
		const int * wcols,  const int64_t * wprods,
		T ** kvs,  const T* vec, const T* grad,
		T ** gkvs) {
	int row = blockIdx.x * blockDim.x + threadIdx.x; // which row of kvs/grad
	T * kvs_other[D-1];
	int wcols_other[D-1];
	int64_t wprods_other[D-1];
	if (row < H) {
		for (int d1 = 0; d1 < D; ++d1) {
			int next_d = 0;
#pragma unroll 
			for (int d2 = 0; d2 < D; ++d2) {
				if (d1 != d2) {
					kvs_other[next_d] = kvs[d2];
					wcols_other[next_d] = wcols[d2];
					wprods_other[next_d] = wprods[d2];
					++next_d;
				}
			}
			int64_t prod_other = 1;
#pragma unroll
			for (int d = 0; d < D-1; ++d) {
				prod_other *= wcols_other[d];
			}
			
			for (int kvs_idx = 0; kvs_idx < wcols[d1]; ++kvs_idx) {
				T outer_tmp = 0.f;
				cu_kvs_dot_vec_inner<T, D-1>(D-1, H, row, wcols_other, wprods_other, 
					kvs_other, vec, &outer_tmp, grad[row], kvs_idx * wprods[d1]);
				gkvs[d1][row + H * (kvs_idx)] = outer_tmp;
			}
		}
	}
}

template <typename T, int D, int WIDTH, int HEIGHT>
__global__ void cu_kvs_dot_vec_vecgrad( int H, int64_t v_len, 
		const int * wcols, const int64_t * wprods, 
		T ** kvs, const T* grad,
		T * gvec) {
	__shared__ T buffer[WIDTH][HEIGHT];
	int idxs[D];
	int64_t vec_idx = threadIdx.y+blockIdx.x*blockDim.y;
	if (vec_idx < v_len) {
		for (int idx = 0; idx < D; ++idx) {
			idxs[idx] = ((vec_idx) / wprods[idx]) % wcols[idx];
		}
		__syncthreads();
		buffer[threadIdx.y][threadIdx.x] = 0;

		// Move down the columns, summing products of kvs into buffer
		for (int row_base = 0; row_base < H; row_base += HEIGHT) {
			if ((row_base + threadIdx.x) < H) {
				T tmp = grad[row_base + threadIdx.x];
	#pragma unroll
				for (int d = 0; d < D; ++d) {
					tmp *= kvs[d][row_base + threadIdx.x + idxs[d] * H];
				}
				buffer[threadIdx.y][threadIdx.x] += tmp;
			}
		}
		__syncthreads();

		// Sum-reduce buffer
		if (threadIdx.x < 32) {
			for (int offset = 32; offset < HEIGHT; offset += 32) {
				buffer[threadIdx.y][threadIdx.x] = buffer[threadIdx.y][threadIdx.x + offset];
			}
		}
		if (threadIdx.x < 16) {
			buffer[threadIdx.y][threadIdx.x] = buffer[threadIdx.y][threadIdx.x + 16];
		}
		if (threadIdx.x < 8) {
			buffer[threadIdx.y][threadIdx.x] = buffer[threadIdx.y][threadIdx.x + 8];
		}
		if (threadIdx.x < 4) {
			buffer[threadIdx.y][threadIdx.x] = buffer[threadIdx.y][threadIdx.x + 4];
		}
		if (threadIdx.x < 2) {
			buffer[threadIdx.y][threadIdx.x] = buffer[threadIdx.y][threadIdx.x + 2];
		}
		if (threadIdx.x == 0) {
			gvec[vec_idx] = buffer[threadIdx.y][0] + buffer[threadIdx.y][1];
		}
	}
}

namespace tensorflow {
	template <typename T, int D>
	void launch_cu_kvs_dot_vec(int H, 
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* d_vec, T* d_out) {
		int threads = 128;
		int nblocks = H / threads + ((H % threads) != 0);
		cu_kvs_dot_vec<T, D><<<nblocks, threads>>>(H, d_wcols, d_wprods, 
								d_kvs, d_vec, d_out);
	}

	template <typename T, int D>
	void launch_cu_kvs_dot_vec_kvsgrad(int H, 
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* d_vec, const T* in_grad,
		T** kvs_grad) {
		int threads = 128;
		int nblocks = H / threads + ((H % threads) != 0);
		cu_kvs_dot_vec_kvsgrad<T, D><<<nblocks, threads>>>(H, d_wcols, d_wprods, 
								d_kvs, d_vec, in_grad, kvs_grad);
	}

	template <typename T, int D>
	void launch_cu_kvs_dot_vec_vecgrad(int H, int64_t vec_len,
		const int * d_wcols, const int64_t * d_wprods,
		T ** d_kvs, const T* in_grad,
		T * vec_grad) {
		const int height = 32;
		const int width = 8;
		dim3 threads{height, width, 1};
		dim3 nblocks{uint32_t(vec_len / width + ((vec_len % width) != 0)), 1, 1 };
		cu_kvs_dot_vec_vecgrad<T, D, height, width><<<nblocks, threads>>>(
				H, vec_len,
				d_wcols, d_wprods, 
				d_kvs, in_grad,
				vec_grad);
	}

}
#define KVS_DOT_VEC_FOR_DIM(T, D) 									\
template void tensorflow::launch_cu_kvs_dot_vec<T, D>(int H, 		\
	const int * d_wcols, const int64_t * d_wprods,					\
	T ** d_kvs, const T * d_vec, T * d_out);						\
template void tensorflow::launch_cu_kvs_dot_vec_kvsgrad<T, D>(int H,\
	const int * d_wcols, const int64_t * d_wprods,					\
	T ** d_kvs, const T* d_vec, const T* in_grad,					\
	T** kvs_grad);													\
template void tensorflow::launch_cu_kvs_dot_vec_vecgrad<T, D>(int H,\
	int64_t vec_len, const int * d_wcols, const int64_t * d_wprods,	\
	T ** d_kvs, const T* in_grad,					\
	T * vec_grad);

KVS_DOT_VEC_FOR_DIM(float, 2)
KVS_DOT_VEC_FOR_DIM(float, 3)
KVS_DOT_VEC_FOR_DIM(float, 4)
KVS_DOT_VEC_FOR_DIM(float, 5)
KVS_DOT_VEC_FOR_DIM(float, 6)
KVS_DOT_VEC_FOR_DIM(float, 7)
KVS_DOT_VEC_FOR_DIM(float, 8)
KVS_DOT_VEC_FOR_DIM(float, 9)
KVS_DOT_VEC_FOR_DIM(float, 10)
KVS_DOT_VEC_FOR_DIM(double, 2)
KVS_DOT_VEC_FOR_DIM(double, 3)
KVS_DOT_VEC_FOR_DIM(double, 4)
KVS_DOT_VEC_FOR_DIM(double, 5)
KVS_DOT_VEC_FOR_DIM(double, 6)
KVS_DOT_VEC_FOR_DIM(double, 7)
KVS_DOT_VEC_FOR_DIM(double, 8)
KVS_DOT_VEC_FOR_DIM(double, 9)
KVS_DOT_VEC_FOR_DIM(double, 10)

#undef KVS_DOT_VEC_FOR_DIM