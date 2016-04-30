#if GOOGLE_CUDA
#define EIGEN_USE_GPU

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/user_ops/gpu_cholgrad_func_gpu.h"
typedef Eigen::GpuDevice GPUDevice;


// Instantiate the GPU implementation for GPU number types.
template struct tensorflow::CholgradHelper<GPUDevice, float>;
// template struct tensorflow::functors::ComputeCholGrad<GPUDevice, double>;


#endif