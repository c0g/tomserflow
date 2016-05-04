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

// See docs in ../ops/linalg_ops.cc.
// TODO(konstantinos): Enable complex inputs. This will require additional tests
//                     and OP_REQUIRES.

#include "tensorflow/core/framework/op.h"
REGISTER_OP("Cholesky")
  .Input("l: T").Output("g: T")
  .Attr("T : {float, double}")
  .Doc("Get cholesky of a square PSD matrix.");
REGISTER_OP("BatchedCholesky")
  .Input("l: T").Output("g: T")
  .Attr("T : {float, double}")
  .Doc("Get cholesky of a batched of square PSD matrices.");

#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/platform/stream_executor.h"

  //WRONG: should include from ../kernels/ but I don't know how to Bazel
#include "tensorflow/core/user_ops/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/user_ops/gpu_cholesky.h"
#include "tensorflow/core/user_ops/cuda_matrix_helper.h"

#include "tensorflow/stream_executor/solver.h"

// #include "cusolverDn.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;



namespace tensorflow {
  namespace functors {
  }

template <typename Device, typename T, bool BatchOp>
class CholeskyOp : public UnaryLinearAlgebraOpBase {
 public:
  explicit CholeskyOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOpBase(context) {}

  TensorShape GetOutputMatrixShape(const TensorShape& input_matrix_shape) override {
    return input_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint64max;
    } else {
      return rows * rows * rows;
    }
  }

  bool SupportsBatchOperation() {
    return BatchOp;
  }
  void ComputeMatrix(OpKernelContext* context, int64 matrix_idx, 
        const Tensor& in, const TensorShape& inshape,
        Tensor* out, const TensorShape& outshape) override {
    OP_REQUIRES(context, inshape.dim_size(0) == in.dim_size(1),
                errors::InvalidArgument("Input matrix must be square."));
    if (inshape.dim_size(0) == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }
    const T* in_sub = in.flat<T>().data() + matrix_idx * inshape.num_elements();
    T* out_sub = out->flat<T>().data() + matrix_idx * outshape.num_elements();
    bool success = true;
    functors::chol_functor<Device, T> chol;
    chol(context, in_sub, inshape.dim_size(0), out_sub, success);
    OP_REQUIRES(context, success,
                    errors::InvalidArgument("LLT decomposition was not successful. "
                                            "The input might not be valid."));
  }
};

namespace {
    template <typename T>
    perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory)
    {
        perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
        perftools::gputools::DeviceMemory<T> typed(wrapped);
        return typed;
    }
    class CusolverScratchAllocator : public perftools::gputools::ScratchAllocator {
     public:
      using Stream = ::perftools::gputools::Stream;
      using DeviceMemoryBytes = ::perftools::gputools::DeviceMemory<uint8>;

      CusolverScratchAllocator(OpKernelContext* context) : context_(context) {}

      int64 GetMemoryLimitInBytes(Stream* stream) override { return -1; }

      perftools::gputools::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
          Stream* stream, int64 byte_size) override {
        Tensor temporary_memory;

        Status allocation_status(context_->allocate_temp(
            DT_UINT8, TensorShape({byte_size}), &temporary_memory));
        if (!allocation_status.ok()) {
          return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
              DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
        }
        // Hold the reference of the allocated tensors until the end of the
        // allocator.
        allocated_tensors_.push_back(temporary_memory);
        return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
            DeviceMemoryBytes::MakeFromByteSize(
                temporary_memory.flat<uint8>().data(),
                temporary_memory.flat<uint8>().size()));
      }

     private:
      OpKernelContext* context_;
      std::vector<Tensor> allocated_tensors_;
    };
} // anonymous namespace
namespace functors {

    template <typename T>
    struct chol_functor<CPUDevice, T> {
      using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using ConstMatrixMap = Eigen::Map<const Matrix>;
      using MatrixMap = Eigen::Map<Matrix>;
      void operator()(OpKernelContext*, const T* in, const int M, T* out, bool& success) {
        //The next three lines are necessary to get Eigen matrix behaviour.
        const ConstMatrixMap in_mat(in, M, M);
        MatrixMap out_mat(out, M, M);
        Eigen::LLT<Matrix> llt_decomposition(in_mat);

        // Output the lower triangular in a dense form.
        out_mat = llt_decomposition.matrixL();
        success = llt_decomposition.info() == Eigen::Success;
      }
    };

    template <typename T>
    struct chol_functor<GPUDevice, T> {
      using Helper = CUDAMatrixHelper<GPUDevice, T>;
      void operator()(OpKernelContext* ctx, const T* in, const int M, T* out, bool& success) {
        Matrix<const T> inMat{in, 0, M, M, M};
        Matrix<T> outMat{out, 0, M, M, M};
        
        auto* stream = ctx->op_device_context()->stream();
        auto outdev = AsDeviceMemory<T>(out);
        // Copy from in to out
        Helper::copy(ctx->eigen_device<GPUDevice>(), outMat, inMat);
        CusolverScratchAllocator scratch_allocator(ctx);
        stream->ThenSolverPotrfWithScratch(perftools::gputools::solver::UpperLower::kUpper, 
                                M, &outdev, M, &scratch_allocator);
        Helper::tril(ctx->eigen_device<GPUDevice>(), outMat);
        stream->BlockHostUntilDone();
        success = true;
      }
  };
} // functors

REGISTER_KERNEL_BUILDER(
    Name("Cholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<CPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("Cholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<CPUDevice, double, false>);

REGISTER_KERNEL_BUILDER(
    Name("Cholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<GPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("Cholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<GPUDevice, double, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchedCholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<CPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchedCholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<CPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchedCholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<GPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchedCholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<GPUDevice, double, true>);

}  // namespace tensorflow
