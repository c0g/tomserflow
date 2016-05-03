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
REGISTER_OP("GPUCholesky")
  .Input("l: T").Output("g: T")
  .Attr("T : {float, double}")
  .Doc("Get cholesky of a square PSD matrix.");

#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/user_ops/gpu_cholesky_func.h"

// #include "cusolverDn.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;



namespace tensorflow {
  namespace functors {

    template <typename T>
    struct chol_functor<CPUDevice, T> {
      using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using ConstMatrixMap = Eigen::Map<const Matrix>;
      using MatrixMap = Eigen::Map<Matrix>;
      void operator()(const CPUDevice& d, const T* in, const int M, T* out, bool& success) {
        //The next three lines are necessary to get Eigen matrix behaviour.
        const ConstMatrixMap in_mat(in, M, M);
        MatrixMap out_mat(out, M, M);
        Eigen::LLT<Matrix> llt_decomposition(in_mat);

        // Output the lower triangular in a dense form.
        out_mat = llt_decomposition.matrixL();
        success = llt_decomposition.info() == Eigen::Success;
      }
    };
  }

template <typename Device, typename T>
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
    return 0;
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
    functors::chol_functor<Device, T> chol;
    bool success = true;
    chol(context->eigen_device<Device>(), in.flat<T>().data(), inshape.dim_size(0), 
      out->flat<T>().data(), success);
    OP_REQUIRES(context, success,
                    errors::InvalidArgument("LLT decomposition was not successful. "
                                            "The input might not be valid."));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("GPUCholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("GPUCholesky")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<CPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("GPUCholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    CholeskyOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("GPUCholesky")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    CholeskyOp<GPUDevice, double>);

}  // namespace tensorflow
