#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/cuda_matrix_helper_impl.h"
typedef Eigen::GpuDevice GPUDevice;

// Instantiate the GPU implementation for GPU number types.
template struct tensorflow::CUDAMatrixHelper<GPUDevice, float>;
template struct tensorflow::CUDAMatrixHelper<GPUDevice, double>;

#endif