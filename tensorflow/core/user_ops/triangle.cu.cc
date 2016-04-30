// /* Copyright 2016 Tom Nickson. All rights reserved. */

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/user_ops/triangle_func_gpu.h"
namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

// Instantiate the GPU implementation for GPU number types.
#define REGISTER_FUNCTORS(type)                           \
  template struct functors::upper_tri<GPUDevice, type>;  \
  template struct functors::lower_tri<GPUDevice, type>;  \
 

REGISTER_FUNCTORS(float);
REGISTER_FUNCTORS(double);
};

#endif