
#include "tensorflow/core/user_ops/gpu_cholesky_func.h"
#include "cusolverDn.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


namespace tensorflow {
	typedef Eigen::GpuDevice GPUDevice;

	cusolverStatus_t cusolver_potrf_bufferSize(cusolverDnHandle_t handle,
                 cublasFillMode_t uplo,
                 int n,
                 float *A,
                 int lda,
                 int *Lwork) {
	  return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork );
	}
	cusolverStatus_t cusolver_potrf_bufferSize(cusolverDnHandle_t handle,
	                 cublasFillMode_t uplo,
	                 int n,
	                 double *A,
	                 int lda,
	                 int *Lwork) {
	  return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork );
	}

	cusolverStatus_t cusolver_potrf(cusolverDnHandle_t handle,
	           cublasFillMode_t uplo,
	           int n,
	           float *A,
	           int lda,
	           float *Workspace,
	           int Lwork,
	           int *devInfo) {
	  return cusolverDnSpotrf( handle, uplo, n, A, lda, Workspace, Lwork, devInfo );
	}
	cusolverStatus_t cusolver_potrf(cusolverDnHandle_t handle,
	           cublasFillMode_t uplo,
	           int n,
	           double *A,
	           int lda,
	           double *Workspace,
	           int Lwork,
	           int *devInfo) {
	  return cusolverDnDpotrf( handle, uplo, n, A, lda, Workspace, Lwork, devInfo );
	}
namespace functors {
    template <typename T>
    struct chol_functor<GPUDevice, T> {
      void operator()(const GPUDevice& d, const T* in, const int M, T* out, bool& success) {
      	// potrf operates in place, need a copy
      	int localM = M;
		cudaMemcpyAsync(out, in, M * M * sizeof(T),
			cudaMemcpyDeviceToDevice, d.stream()); 
		cusolverDnHandle_t handle;
		if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
			success = false;
			std::cout << "*****Could not create handle*****" << std::endl;
			return;
		}
		if (cusolverDnSetStream(handle, d.stream()) != CUSOLVER_STATUS_SUCCESS) {
			success = false;
			std::cout << "****Could not set stream*****" << std::endl;
			return;
		}
		int wsize;
		T * wspace;
		// cusolver is col-major, so UPPER gives us lower cholesky
		cusolver_potrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER,  
		        localM, out, localM, &wsize ); 

		// Allocate scratch space
		cudaMalloc(&wspace, sizeof(T) * wsize);

		int devinfo;
		cusolver_potrf(handle, CUBLAS_FILL_MODE_UPPER, 
		        localM, out, localM, wspace, wsize, &devinfo);

		// tidy up
		cudaFree(wspace);
		cusolverDnDestroy(handle);
      }
    };
}
}