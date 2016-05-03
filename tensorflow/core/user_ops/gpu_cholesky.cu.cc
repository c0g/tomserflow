// /* Copyright 2016 Tom Nickson. All rights reserved. */

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/user_ops/gpu_cholesky_func_gpu.h"
typedef Eigen::GpuDevice GPUDevice;


// Instantiate the GPU implementation for GPU number types.
template struct tensorflow::functors::chol_functor<GPUDevice, float>;
template struct tensorflow::functors::chol_functor<GPUDevice, double>;


#endif