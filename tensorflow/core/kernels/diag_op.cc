/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#define EIGEN_USE_THREADS
// See docs in ../ops/array_ops.cc
#include "tensorflow/core/kernels/diag_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/framework/types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
namespace tensorflow {
  namespace functor {

    // Partial specialization of SetZeroFunctor<Device=CPUDevice, T>.
    template <typename T>
    struct SetZeroFunctor<CPUDevice, T> {
      void operator()(const CPUDevice& d, typename TTypes<T>::Flat out) {
        out.device(d) = out.constant(T());
      }
    };

    template<typename T>
    struct SetDiag<CPUDevice, T> {
        void operator()(const CPUDevice& d, 
          uint64 N, const T* diag, T* tensor) {
            for (uint64 idx = 0; idx < N; ++idx) {
              uint64 tidx = idx * (N + 1);
              tensor[tidx] = diag[idx];
            }
        }
    };
    template<typename T>
    struct GetDiag<CPUDevice, T> {
        void operator()(const CPUDevice& d, 
          uint64 N, const T* tensor, T* diag) {
          for (uint64 idx = 0; idx < N; ++ idx) {
            uint64 tidx = idx * (N + 1);
            diag[idx] = tensor[tidx];
          }
        }
    };
  } // functor

// Generate the diagonal tensor with the diagonal set to the input tensor.
// It only allows up to rank 3 input tensor, so the output tensor is up to
// rank 6.
template <typename Device, typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(context, 1 <= num_dims && num_dims <= 3,
                errors::InvalidArgument("Expected 1 <= dims <= 3, got shape ",
                                        diagonal.shape().DebugString()));
    uint64 N = 1;
    for (uint64 idx = 0; idx < diagonal.dims(); ++idx){
      N *= diagonal.dim_size(idx);
    }
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));

    const T* diag_ptr = diagonal.flat<T>().data();
    T* tensor_ptr = output_tensor->flat<T>().data();
    functor::SetDiag<Device, T> diag;
    diag(context->eigen_device<Device>(), N, diag_ptr, tensor_ptr);
  }
};

#define REGISTER_DIAGOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), DiagOp<CPUDevice, T>);
REGISTER_DIAGOP(double);
REGISTER_DIAGOP(float);
REGISTER_DIAGOP(int32);
REGISTER_DIAGOP(int64);

#undef REGISTER_DIAGOP

#define REGISTER_DIAGOP_GPU(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("Diag").Device(DEVICE_GPU).TypeConstraint<T>("T"), DiagOp<GPUDevice, T>);
REGISTER_DIAGOP_GPU(double);
REGISTER_DIAGOP_GPU(float);
REGISTER_DIAGOP_GPU(int32);
REGISTER_DIAGOP_GPU(int64);

#undef REGISTER_DIAGOP_GPU


// Generate the diagonal tensor with the diagonal set to the input tensor.
// It only allows rank 2, 4, or 6 input tensor, so the output tensor is 
// rank 1, 2, or 3.
template <typename Device, typename T>
class DiagPartOp : public OpKernel {
    public:
 explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(context, 2 == num_dims || 4 == num_dims || 6 == num_dims, 
                errors::InvalidArgument("The rank of the tensor should be 2, \
                                         4, or 6, got shape ",
                                        tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++){
      OP_REQUIRES(context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
                  errors::InvalidArgument(
                    "Invalid shape ", tensor.shape().DebugString(),
                    ": dimensions ", i, " and ", i + out_dims, " do not match.")
                  );
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));

    uint64 N = 1;
    for (uint64 idx = 0; idx < out_dims; ++idx) {
      N *= out_shape.dim_size(idx);
    }
    const T* tensor_ptr = tensor.flat<T>().data();
    T* diag_ptr = output->flat<T>().data();
    functor::GetDiag<Device, T> diag;
    diag(context->eigen_device<Device>(), N, tensor_ptr, diag_ptr);
  }
};

#define REGISTER_DIAGPARTOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("DiagPart").Device(DEVICE_CPU).TypeConstraint<T>("T"), DiagPartOp<CPUDevice, T>);

REGISTER_DIAGPARTOP(double);
REGISTER_DIAGPARTOP(float);
REGISTER_DIAGPARTOP(int32);
REGISTER_DIAGPARTOP(int64);

#undef REGISTER_DIAGPARTOP


#define REGISTER_GPU_DIAGPARTOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("DiagPart").Device(DEVICE_GPU).TypeConstraint<T>("T"), DiagPartOp<GPUDevice, T>);

REGISTER_GPU_DIAGPARTOP(double);
REGISTER_GPU_DIAGPARTOP(float);
REGISTER_GPU_DIAGPARTOP(int32);
REGISTER_GPU_DIAGPARTOP(int64);

#undef REGISTER_GPU_DIAGPARTOP

}  // namespace tensorflow
