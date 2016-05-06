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
#include <cmath>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/binary_linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif // GOOGLE_CUDA

namespace tensorflow {
#if GOOGLE_CUDA
namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory)
{
    perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
    perftools::gputools::DeviceMemory<T> typed(wrapped);
    return typed;
}
}//namespace

#endif  // GOOGLE_CUDA

template <class Scalar, bool SupportsBatchOperationT>
class MatrixTriangularSolveOp
    : public BinaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* context)
      : BinaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context),
        lower_(true),
        adjoint_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }
  ~MatrixTriangularSolveOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape,
      const TensorShape& rhs_matrix_shape) override {
    CHECK_EQ(input_matrix_shape.dims(), rhs_matrix_shape.dims());
    TensorShape output_matrix_shape = input_matrix_shape;
    output_matrix_shape.set_dim(
        output_matrix_shape.dims() - 1,
        rhs_matrix_shape.dim_size(output_matrix_shape.dims() - 1));
    return output_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape,
                       const TensorShape& rhs_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 rhss = rhs_matrix_shape.dim_size(1);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return rows * rows * rhss;
    }
  }

  using typename BinaryLinearAlgebraOp<Scalar,
                                       SupportsBatchOperationT>::MatrixMap;
  using typename BinaryLinearAlgebraOp<Scalar,
                                       SupportsBatchOperationT>::ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& matrix,
                     const ConstMatrixMap& rhs, MatrixMap* output) override {
    OP_REQUIRES(context, matrix.rows() == matrix.cols(),
                errors::InvalidArgument("Input matrix must be square."));
    OP_REQUIRES(
        context, matrix.cols() == rhs.rows(),
        errors::InvalidArgument("Input matrix and rhs are incompatible."));
    if (matrix.rows() == 0 || rhs.cols() == 0) {
      // To be consistent with the MatrixInverse op, we define the solution for
      // an empty set of equation as the empty matrix.
      return;
    }
    const Scalar min_abs_pivot = matrix.diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > Scalar(0),
                errors::InvalidArgument("Input matrix is not invertible."));
    if (lower_) {
      auto triangle = matrix.template triangularView<Eigen::Lower>();
      if (adjoint_) {
        output->noalias() = triangle.adjoint().solve(rhs);
      } else {
        output->noalias() = triangle.solve(rhs);
      }
    } else {
      auto triangle = matrix.template triangularView<Eigen::Upper>();
      if (adjoint_) {
        output->noalias() = triangle.adjoint().solve(rhs);
      } else {
        output->noalias() = triangle.solve(rhs);
      }
    }
  }

 private:
  bool lower_;
  bool adjoint_;

  TF_DISALLOW_COPY_AND_ASSIGN(MatrixTriangularSolveOp);
};

#ifdef GOOGLE_CUDA
template <class T, bool SupportsBatchOperationT>
class MatrixTriangularSolveOpGPU
    : public BinaryLinearAlgebraOpBase {
 public:
  explicit MatrixTriangularSolveOpGPU(OpKernelConstruction* context)
      : BinaryLinearAlgebraOpBase(context),
        lower_(true),
        adjoint_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }
  ~MatrixTriangularSolveOpGPU() override {}

  bool SupportsBatchOperation() {
    return SupportsBatchOperationT;
  }

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape,
      const TensorShape& rhs_matrix_shape) override {
    CHECK_EQ(input_matrix_shape.dims(), rhs_matrix_shape.dims());
    TensorShape output_matrix_shape = input_matrix_shape;
    output_matrix_shape.set_dim(
        output_matrix_shape.dims() - 1,
        rhs_matrix_shape.dim_size(output_matrix_shape.dims() - 1));
    return output_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape,
                       const TensorShape& rhs_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 rhss = rhs_matrix_shape.dim_size(1);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return rows * rows * rhss;
    }
  }
  using UpLo = perftools::gputools::blas::UpperLower;
  using Trans = perftools::gputools::blas::Transpose;
  void ComputeMatrix(OpKernelContext* context, int64 matrix_index,
                             const Tensor& a_in,
                             const TensorShape& a_in_matrix_shape,
                             const Tensor& b_in,
                             const TensorShape& b_in_matrix_shape,
                             Tensor* output,
                             const TensorShape& output_matrix_shape) override {
    OP_REQUIRES(context, a_in_matrix_shape.dim_size(0) 
                    == a_in_matrix_shape.dim_size(1),
                errors::InvalidArgument("Input matrix must be square."));
    OP_REQUIRES(
        context, a_in_matrix_shape.dim_size(1) 
                    == b_in_matrix_shape.dim_size(0),
        errors::InvalidArgument("Input matrix and rhs are incompatible."));
    auto aptr = AsDeviceMemory(a_in.flat<T>().data() +
             matrix_index * a_in_matrix_shape.num_elements());
    auto bptr = AsDeviceMemory(b_in.flat<T>().data() +
             matrix_index * b_in_matrix_shape.num_elements());
    auto cptr = AsDeviceMemory(output->flat<T>().data() +
             matrix_index * output_matrix_shape.num_elements());

    auto* stream = context->op_device_context()->stream();

    // Copy b into output (cublas works inplace)
    uint64 belems = b_in_matrix_shape.num_elements();
    stream->ThenMemcpyD2D(&cptr, bptr, sizeof(T) * belems);

    UpLo uplo;
    Trans trans;
    if (lower_) {
      uplo = UpLo::kLower;
    } else {
      uplo = UpLo::kUpper;
    }
    if (adjoint_) {
      trans = Trans::kTranspose;
    } else {
      trans = Trans::kNoTranspose;
    }
    uint64 lda = a_in_matrix_shape.dim_size(1);
    uint64 ldb = b_in_matrix_shape.dim_size(1);
    uint64 cublas_m = b_in_matrix_shape.dim_size(1);
    uint64 cublas_n = b_in_matrix_shape.dim_size(0);
    stream->ThenBlasTrsm(
      perftools::gputools::blas::Side::kRight,
      uplo, trans, 
      perftools::gputools::blas::Diagonal::kNonUnit,
      cublas_m, cublas_n, 1.0, 
      aptr, lda,
      &bptr, ldb
    );
    stream->BlockHostUntilDone();
  }

 private:
  bool lower_;
  bool adjoint_;

  TF_DISALLOW_COPY_AND_ASSIGN(MatrixTriangularSolveOpGPU);
};
#endif

REGISTER_BINARY_LINALG_OP("MatrixTriangularSolve",
                          (MatrixTriangularSolveOp<float, false>), float);
REGISTER_BINARY_LINALG_OP("MatrixTriangularSolve",
                          (MatrixTriangularSolveOp<double, false>), double);
REGISTER_BINARY_LINALG_OP("BatchMatrixTriangularSolve",
                          (MatrixTriangularSolveOp<float, true>), float);
REGISTER_BINARY_LINALG_OP("BatchMatrixTriangularSolve",
                          (MatrixTriangularSolveOp<double, true>), double);

REGISTER_KERNEL_BUILDER(
    Name("MatrixTriangularSolve")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    MatrixTriangularSolveOpGPU<float, false>);

REGISTER_KERNEL_BUILDER(
    Name("MatrixTriangularSolve")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    MatrixTriangularSolveOpGPU<double, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMatrixTriangularSolve")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    MatrixTriangularSolveOpGPU<float, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMatrixTriangularSolve")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    MatrixTriangularSolveOpGPU<double, true>);

}  // namespace tensorflow
