#pragma once

#include "tensorflow/core/user_ops/gpu_cholgrad_func.h"

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
            cudaMemcpyAsync(dst.data(), src.data(), src.m * src.n * sizeof(T),
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
    };
} // tensorflow


// #pragma once
// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/tensor_types.h"
// #include "tensorflow/core/user_ops/gpu_cholgrad_func.h"

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/platform/stream_executor.h"

// typdef GPUDevice Eigen::GpuDevice;

// #if GOOGLE_CUDA

// // #include "tensorflow/stream_executor/stream.h"

// namespace tensorflow {
//     using Stream = perftools::gputools::Stream;
// template <typename T>
// perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory)
// {
//     perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
//     perftools::gputools::DeviceMemory<T> typed(wrapped);
//     return typed;
// }
// template <typename T>
// struct Matrix {
//     T* dataptr;
//     int offset; // offset into parent
//     int m; // height of block
//     int n; // width of block
//     int ld; // row major leading dim of parent
//     T* data() { return dataptr + offset; }
//     Matrix view(int r0, int r1, int c0, int c1);
// };
// template <typename T>
// Matrix<T> Matrix<T>::view(int r0, int r1, int c0, int c1)
// {
//     r1 = (r1 == -1) ? m : r1;
//     c1 = (c1 == -1) ? n : c1;
//     int newm = r1 - r0;
//     int newn = c1 - c0;
//     int newoffset = r0 * ld + c0;
//     return Matrix<T>{ data(), newoffset, newm, newn, ld };
// }
// template <typename T>
// struct L3Par {
//     Matrix<T> R;
//     Matrix<T> D;
//     Matrix<T> B;
//     Matrix<T> C;
//     L3Par(Matrix<T>& parent, int j, int k);
// };
// template <typename T>
// L3Par<T>::L3Par<T>(Matrix<T>& parent, int j, int k)
// {
//     R = parent.view(j, k, 0, j);
//     D = parent.view(j, k, j, k);
//     B = parent.view(k, -1, 0, j);
//     C = parent.view(k, -1, j, k);
// }

// // wrapper to rearrange arguments for cuBLAS
// // cublas does C = AB where A,B,C are column major
// // for row major matrices, use C' = B'A'
// template <typename T>
// void gemm(Stream* stream,
//     bool transa, bool transb,
//     T alpha, Matrix<const T>& A, Matrix<const T>& B,
//     T beta, Matrix<T>& C)
// {
//     auto aptr = AsDeviceMemory(A.data());
//     auto bptr = AsDeviceMemory(B.data());
//     auto cptr = AsDeviceMemory(C.data());

//     uint64 m = transb ? B.m : B.n;
//     uint64 k = transb ? B.n : B.m;
//     uint64 n = transa ? A.n : A.m;
//     int lda = A.ld;
//     int ldb = B.ld;
//     int ldc = C.ld;

//     perftools::gputools::blas::Transpose real_transa = transa ? 
//         perftools::gputools::blas::Transpose::kNoTranspose : 
//         perftools::gputools::blas::Transpose::kTranspose;
//     perftools::gputools::blas::Transpose real_transb = transb ? 
//         perftools::gputools::blas::Transpose::kNoTranspose : 
//         perftools::gputools::blas::Transpose::kTranspose;

//     stream->ThenBlasGemm(
//         real_transb, real_transa,
//         m, n, k,
//         alpha,
//         bptr, ldb,
//         aptr, lda,
//         beta,
//         &cptr, ldc);
// }

// // wrapper to rearrange arguments for cuBLAS
// // cublas does: C = op(A) \ B or C = B / op(A),
// // where op may be transpose or no-transpose.
// // For each config, we change as the following table:
// /*   row major  | col major
//     ------------+------------
//     C = A \ B   |  C' = B' / A'
//     C = B / A   |  C' = A' \ B'
//     C = B / A'  |  C' = A \ B'
//     C = A' \ B  |  C' = B' / A' 
// */
// // Rule: switch sides A,B, permute transpose.
// // uplo: switch 'L' to 'U' and vice versa?

// template <typename T>
// void trsm(Stream* stream,
//     char side, char uplo,
//     bool transa, T alpha,
//     Matrix<const T>& A, Matrix<T>& B)
// {
//     auto aptr = AsDeviceMemory(A.data());
//     auto bptr = AsDeviceMemory(B.data());

//     uint64 m = transa ? A.m : A.n;
//     uint64 n = transa ? A.n : A.m;
//     perftools::gputools::blas::Transpose real_transa = transa ? 
//         perftools::gputools::blas::Transpose::kNoTranspose : 
//         perftools::gputools::blas::Transpose::kTranspose;

//     perftools::gputools::blas::Side real_side;
//     switch (side) {
//     case 'L':
//         real_side = perftools::gputools::blas::Side::kRight;
//         break;
//     case 'R':
//         real_side = perftools::gputools::blas::Side::kLeft;
//         break;
//     }
//     perftools::gputools::blas::UpperLower real_uplo;
//     switch (uplo) {
//     case 'U':
//         real_uplo = perftools::gputools::blas::UpperLower::kLower;
//         break;
//     case 'L':
//         real_uplo = perftools::gputools::blas::UpperLower::kUpper;
//         break;
//     }
//     perftools::gputools::blas::Diagonal diagonal = perftools::gputools::blas::Diagonal::kNonUnit;

//     stream->ThenBlasTrsm(
//         real_side, real_uplo,
//         real_transa, diagonal,
//         m, n, alpha,
//         aptr, A.ld,
//         &bptr, B.ld);
// }

// // Copies lower triangle into upper triangle
// // ignores diagonal
// __global__ void cu_symmetrise(float* data, int m, int ld)
// {
//     int r = blockDim.y * blockIdx.y + threadIdx.y;
//     int c = blockDim.x * blockIdx.x + threadIdx.x;
//     if (c < m) {
//         if (r < c) {
//             // r, c point into upper triangle exluding diag
//             int uidx = c * ld + r;
//             int lidx = r * ld + c;
//             data[uidx] = data[lidx];
//         }
//     }
// }
// template <typename T>
// void symmetrise(cudaStream_t& stream, Matrix<T>& mat)
// {
//     assert(mat.m == mat.n);
//     dim3 blocksize(16, 16);
//     dim3 nblocks(mat.m / 16 + 1, mat.n / 16 + 1);
//     cu_symmetrise<<<nblocks, blocksize, 0, stream>>>(mat.data(), mat.m, mat.ld);
// }

// // Zeros upper triangle (above diagonal)
// __global__ void cu_tril(float* data, int m, int ld)
// {
//     int r = blockDim.y * blockIdx.y + threadIdx.y;
//     int c = blockDim.x * blockIdx.x + threadIdx.x;
//     if (c < m) {
//         if (r < c) {
//             int idx = c * ld + r;
//             data[idx] = 0;
//         }
//     }
// }
// template <typename T>
// void tril(cudaStream_t& stream, Matrix<T>& mat)
// {
//     assert(mat.m == mat.n);
//     dim3 blocksize(16, 16);
//     dim3 nblocks(mat.m / 16 + 1, mat.n / 16 + 1);
//     cu_tril<<<nblocks, blocksize, 0, stream>>>(mat.data(), mat.m, mat.ld);
// }
// //Halves diagonal
// __global__ void cu_half_d(float* data, int m, int ld)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int didx = idx * (ld + 1);
//     int last_idx = m * (ld + 1);
//     if (didx < last_idx) {
//         data[didx] *= 0.5;
//     }
// }
// // Lower tril then halves diagonal
// template <typename T>
// void phi(cudaStream_t& stream, Matrix<T>& mat)
// {
//     assert(mat.m == mat.n);
//     tril(stream, mat);
//     int blocksize = 256;
//     int nblocks = mat.m / 256 + 1;
//     // std::cout << nblocks << " block of size " << blocksize << std::endl;
//     cu_half_d<<<nblocks, blocksize, 0, stream>>>(mat.data(), mat.m, mat.ld);
// }

// namespace functors {
//     template <typename T>
//     struct ComputeCholGrad<GPUDevice, T> {
//         const int blocksize = 1024;

//         void operator()(OpKernelContext* ctx, const Tensor& Ltensor, const Tensor& Ltensorbar, Tensor* Atensorbar)
//         {
//             auto cudaStream = ctx->eigen_device<GPUDevice>().stream();
//             auto* stream = ctx->op_device_context()->stream();
//             const T* Lptr = Ltensor.flat<T>().data();
//             const T* Lbarptr = Ltensorbar.flat<T>().data();
//             T* Abarptr = Atensorbar->flat<T>().data();
//             int M = Ltensor.dim_size(0);
//             Matrix<const T> L{ Lptr, 0, M, M, M };
//             Matrix<const T> Lbar{ Lbarptr, 0, M, M, M };
//             Matrix<T> Abar{ Abarptr, 0, M, M, M };

//             // Copy Lbar into Abar on GPUStream
//             cudaMemcpyAsync(Abar.data(), Lbar.data(), M * M * sizeof(T),
//                 cudaMemcpyDeviceToDevice, cudaStream);

//             // Allocate scratch space (blocksize)
//             Tensor* scratchtensor = nullptr;
//             TensorShape tmp_shape{ blocksize, blocksize };
//             ctx->allocate_temp(DataTypeToEnum<T>::value, tmp_shape, scratchtensor);
//             T* scratchptr = scratchtensor->flat<T>().data();
//             Matrix<T> scratch{ scratchptr, 0, blocksize, blocksize, blocksize };

//             CholeskyGradSymbolic(ctx, L, Lbar, Abar);
//         }
//         void CholeskyGradSymbolic(OpKernelContext* ctx, Matrix<const T>& L, Matrix<const T>& Lbar, Matrix<T>& Abar)
//         {
//             auto* stream = ctx->op_device_context()->stream();
//             auto cudaStream = ctx->eigen_device<GPUDevice>().stream();
//             // P <- L^T Lbar
//             T one = 1.0;
//             T zero = 0.0;
//             gemm(stream, true, false, one, L, Lbar, zero, Abar);
//             stream->ok();
//             // P <- Phi(L^-T Lbar) + Phi(L^-T Lbar)^T
//             symmetrise(cudaStream, Abar);
//             stream->ok();
//             // P <- L^-T(Phi(L^-T Lbar) + Phi(L^-T Lbar)^T)
//             trsm(stream, 'L', 'L', true, one, L, Abar);
//             // P <- L^-T(Phi(L^-T Lbar) + Phi(L^-T Lbar)^T)
//             trsm(stream, 'R', 'L', true, one, L, Abar);
//             stream->ok();
//             // P <- Phi(L^-T(Phi(L^-T Lbar) + Phi(L^-T Lbar)^T) L^-1)
//             phi(cudaStream, Abar);
//         }
//     };
// } // namespace functors
// } // namespace tensorflow
// #endif // GOOGLE_CUDA