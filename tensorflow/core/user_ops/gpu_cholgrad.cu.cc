#if GOOGLE_CUDA
#define EIGEN_USE_GPU

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/user_ops/cuda_matrix_helper_impl.h"
typedef Eigen::GpuDevice GPUDevice;

// Instantiate the GPU implementation for GPU number types.
template struct tensorflow::CUDAMatrixHelper<GPUDevice, float>;
template struct tensorflow::CUDAMatrixHelper<GPUDevice, double>;

#endif