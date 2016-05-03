#pragma once

#include "tensorflow/core/user_ops/gpu_cholgrad.h"

namespace tensorflow {

	namespace {
		// Copies lower triangle into upper triangle
		// ignores diagonal
		template <typename T>
		__global__ void cu_symmetrise(T* data, int m, int ld)
		{
		    int r = blockDim.y * blockIdx.y + threadIdx.y;
		    int c = blockDim.x * blockIdx.x + threadIdx.x;
		    if (c < m) {
		        if (r < c) {
		            // r, c point into upper triangle exluding diag
		            int uidx = r * ld + c;
		            int lidx = c * ld + r;
		            data[uidx] = data[lidx];
		        }
		    }
		}
		// Halves non-diagonal elements
		template <typename T>
		__global__ void cu_half_nd(T* data, int m, int ld)
		{
		    int r = blockDim.y * blockIdx.y + threadIdx.y;
		    int c = blockDim.x * blockIdx.x + threadIdx.x;
		    if ((c < m) && (r < m)) {
		        if (r != c) {
		            int idx = c * ld + r;
		            data[idx] *= 0.5;
		        }
		    }
		}
		// Zeros upper triangle (above diagonal)
		template <typename T>
		__global__ void cu_tril(T* data, int m, int ld)
		{
		    int r = blockDim.y * blockIdx.y + threadIdx.y;
		    int c = blockDim.x * blockIdx.x + threadIdx.x;
		    if (c < m) {
		        if (r < c) {
		            int idx = r * ld + c;
		            data[idx] = 0;
		        }
		    }
		}
		//Halves diagonal
		template <typename T>
		__global__ void cu_half_d(T* data, int m, int ld)
		{
		    int idx = blockDim.x * blockIdx.x + threadIdx.x;
		    int didx = idx * (ld + 1);
		    int last_idx = m * (ld + 1);
		    if (didx < last_idx) {
		        data[didx] *= 0.5;
		    }
		}

	} // anonymous namespace


    typedef Eigen::GpuDevice GPUDevice;

    template <typename T>
    struct CholgradHelper<GPUDevice, T> {
        static void copy(const GPUDevice& d, Matrix<T> dst, Matrix<const T> src) {
            // cudaMemcpyAsync(dst.data(), src.data(), src.m * src.n * sizeof(T),
            //     cudaMemcpyDeviceToDevice, d.stream());
            cudaMemcpy2DAsync(dst.data(),
                        sizeof(T) * dst.m,
                        src.data(), 
                        sizeof(T) * src.ld,
                        sizeof(T) * src.m, src.n,
                        cudaMemcpyDeviceToDevice, d.stream());
        }
        static void symmetrise(const GPUDevice& d, Matrix<T> mat) {
        	dim3 blocksize(16, 16);
		    dim3 nblocks(mat.m / 16 + 1, mat.n / 16 + 1);
		    cu_symmetrise<<<nblocks, blocksize, 0, d.stream()>>>(mat.data(), mat.m, mat.ld);
        }
        static void tril(const GPUDevice&d, Matrix<T> mat) {
        	dim3 blocksize(16, 16);
		    dim3 nblocks(mat.m / 16 + 1, mat.n / 16 + 1);
		    cu_tril<<<nblocks, blocksize, 0, d.stream()>>>(mat.data(), mat.m, mat.ld);
        }
        static void phi(const GPUDevice& d, Matrix<T> mat) {
			tril(d, mat);
		    int blocksize = 256;
		    int nblocks = mat.m / 256 + 1;
		    cu_half_d<<<nblocks, blocksize, 0, d.stream()>>>(mat.data(), mat.m, mat.ld);
        }
        static void reflect_half(const GPUDevice& d, Matrix<T> mat) {
        	dim3 blocksize(16, 16);
		    dim3 nblocks(mat.m / 16 + 1, mat.n / 16 + 1);
		    cu_symmetrise<<<nblocks, blocksize, 0, d.stream()>>>(mat.data(), mat.m, mat.ld);
		    cu_half_nd<<<nblocks, blocksize, 0, d.stream()>>>(mat.data(), mat.m, mat.ld);
        }
    };
} // tensorflow