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

#include "tensorflow/core/framework/op.h"
REGISTER_OP("CholeskyGrad")
    .Input("l: T")
    .Input("lbar: T")
    .Output("abar: T")
    .Attr("T: {float, double}")
    .Doc("Gradients from Lbar pushed back to Abar");

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif // GOOGLE_CUDA


#include "tensorflow/core/user_ops/cuda_matrix_helper.h"
#include <algorithm>
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef perftools::gputools::Stream Stream;

namespace functors {
    template <typename Device, typename T>
    struct ComputeCholGrad {
        void operator()(OpKernelContext* ctx, const Tensor& Ltensor, const Tensor& Ltensorbar, Tensor* Atensorbar);
    };
}

template <typename Device, typename T>
class CholeskyGrad : public OpKernel {
public:
    explicit CholeskyGrad(OpKernelConstruction* context)
        : OpKernel(context){};
    void Compute(OpKernelContext* context) override
    {
        const Tensor& Ltensor = context->input(0);
        const Tensor& Lbartensor = context->input(1);
        // Check that input tensors represent a matrix.
        OP_REQUIRES(context, TensorShapeUtils::IsMatrix(Ltensor.shape()),
            errors::InvalidArgument("In[0] is not a matrix"));
        OP_REQUIRES(context, TensorShapeUtils::IsMatrix(Lbartensor.shape()),
            errors::InvalidArgument("In[1] is not a matrix"));
        // Check that input tensors are square.
        OP_REQUIRES(context,
            Ltensor.dim_size(0) == Ltensor.dim_size(1),
            errors::InvalidArgument("Input matrix must be square."));
        OP_REQUIRES(context,
            Lbartensor.dim_size(0) == Lbartensor.dim_size(1),
            errors::InvalidArgument("Input matrix must be square."));

        // Check that input tensors are of same size.
        OP_REQUIRES(context,
            Ltensor.dim_size(0) == Lbartensor.dim_size(0),
            errors::InvalidArgument("Input matrices must be same size."));

        // Create an output tensor
        Tensor* Abartensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, Lbartensor.shape(), &Abartensor));

        if (Abartensor->NumElements() == 0) {
            // the output shape is a 0-element matrix, so there is nothing to do.
            return;
        }
        functors::ComputeCholGrad<Device, T> cholgrad;
        cholgrad(context, Ltensor, Lbartensor, Abartensor);
    }
};

namespace functors {
        template <typename T>
    struct ComputeCholGrad<CPUDevice, T> {
        using Matrix =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using ConstMatrixMap = Eigen::Map<const Matrix>;
        using MatrixMap = Eigen::Map<Matrix>;
        using ConstRef = Eigen::Ref<const Matrix>;
        using Ref = Eigen::Ref<Matrix>;
        void operator()(OpKernelContext* ctx, const Tensor& Ltensor, const Tensor& Ltensorbar, Tensor* Atensorbar)
        {
            // The next lines are necessary to get Eigen matrix behaviour.
            const ConstMatrixMap input_matrix_l_full(Ltensor.flat<T>().data(),
                Ltensor.dim_size(0),
                Ltensor.dim_size(1));
            const ConstMatrixMap input_matrix_grad(Ltensorbar.flat<T>().data(),
                Ltensor.dim_size(0),
                Ltensor.dim_size(1));
            MatrixMap output_matrix(Atensorbar->template flat<T>().data(),
                Ltensor.dim_size(0),
                Ltensor.dim_size(1));

            // Algorithm only depends on lower triangular half on input_tensor_l.
            const Matrix input_matrix_l = input_matrix_l_full.template triangularView<Eigen::Lower>();
            // Algorithm only depends on lower triangular half on input_matrix_grad.
            output_matrix = input_matrix_grad.template triangularView<Eigen::Lower>();

            const int64 kMatrixSize = input_matrix_l.rows();
            const int64 kMaxBlockSize = 32;

            for (int64 block_end = kMatrixSize; block_end > 0;
                 block_end -= kMaxBlockSize) {
                /* This shows the block structure.

                  /      \
                  |      |
                  | R D  |
                  \ B C  /

                  Variables names representing the derivative matrix have a trailing '_bar'.
                */

                const int64 block_begin = std::max(0ll, block_end - kMaxBlockSize);
                const int64 block_size = block_end - block_begin;
                const int64 trailing_size = kMatrixSize - block_end;

                auto B = input_matrix_l.block(block_end, 0, trailing_size, block_begin);
                auto B_bar = output_matrix.block(block_end, 0, trailing_size, block_begin);

                auto C = input_matrix_l.block(block_end, block_begin, trailing_size,
                    block_size);
                auto C_bar = output_matrix.block(block_end, block_begin, trailing_size,
                    block_size);

                auto D = input_matrix_l.block(block_begin, block_begin, block_size,
                    block_size);
                auto D_bar = output_matrix.block(block_begin, block_begin, block_size, block_size);

                auto R = input_matrix_l.block(block_begin, 0, block_size, block_begin);
                auto R_bar = output_matrix.block(block_begin, 0, block_size, block_begin);

                C_bar = D.adjoint().template triangularView<Eigen::Upper>().solve(C_bar.adjoint()).adjoint();
                D_bar -= (C_bar.adjoint() * C).template triangularView<Eigen::Lower>();
                B_bar -= C_bar * R;
                R_bar -= C_bar.adjoint() * B;
                CholeskyGradUnblocked(D, D_bar);
                R_bar -= (D_bar + D_bar.adjoint()) * R;
            }
            output_matrix = (0.5 * (output_matrix + output_matrix.transpose())).eval();
        }
        void CholeskyGradUnblocked(const ConstRef l_block, Ref grad_block)
        {
            const int64 kMatrixSize = l_block.rows();
            for (int64 k = kMatrixSize - 1; k >= 0; k--) {
                /* This shows the block structure.

                  /      \
                  |      |
                  | r d  |
                  \ B c  /

                  Variables names representing the derivative matrix have a trailing '_bar'.
                */

                const int64 number_rows_B = kMatrixSize - (k + 1);
                const int64 number_rows_r_stack_B = number_rows_B + 1;

                auto r = l_block.block(k, 0, 1, k);
                auto r_bar = grad_block.block(k, 0, 1, k);
                auto d = l_block(k, k); // This needs to be a scalar rather than a view.
                auto d_bar = grad_block.block(k, k, 1, 1);
                // B is not included explicitly because it is not used on its own.
                auto B_bar = grad_block.block(k + 1, 0, number_rows_B, k);
                auto c = l_block.block(k + 1, k, number_rows_B, 1);
                auto c_bar = grad_block.block(k + 1, k, number_rows_B, 1);
                // Result of vertical stacking d_bar and c_bar.
                auto d_stack_c_bar = grad_block.block(k, k, number_rows_r_stack_B, 1);
                // Result of vertical stacking of r and B.
                auto r_stack_B = l_block.block(k, 0, number_rows_r_stack_B, k);
                d_bar -= (c.adjoint() * c_bar) / d;
                d_stack_c_bar /= d;
                r_bar -= d_stack_c_bar.adjoint() * r_stack_B;
                B_bar -= c_bar * r;
                d_bar /= 2.;
            }
        }
    };
}

#ifdef GOOGLE_CUDA
namespace {
    template <typename T>
    perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory)
    {
        perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
        perftools::gputools::DeviceMemory<T> typed(wrapped);
        return typed;
    }

    // wrapper to rearrange arguments for cuBLAS
    // cublas does C = AB where A,B,C are column major
    // for row major matrices, use C' = B'A'
    template <typename T>
    void gemm(Stream* stream,
        bool transa, bool transb,
        T alpha, Matrix<const T>& A, Matrix<const T>& B,
        T beta, Matrix<T>& C)
    {
        auto aptr = AsDeviceMemory(A.data());
        auto bptr = AsDeviceMemory(B.data());
        auto cptr = AsDeviceMemory(C.data());

        uint64 m = transb ? B.m : B.n;
        uint64 k = transb ? B.n : B.m;
        uint64 n = transa ? A.n : A.m;
        int lda = A.ld;
        int ldb = B.ld;
        int ldc = C.ld;

        perftools::gputools::blas::Transpose real_transa = transa ? perftools::gputools::blas::Transpose::kTranspose : perftools::gputools::blas::Transpose::kNoTranspose;
        perftools::gputools::blas::Transpose real_transb = transb ? perftools::gputools::blas::Transpose::kTranspose : perftools::gputools::blas::Transpose::kNoTranspose;

        stream->ThenBlasGemm(
            real_transb, real_transa,
            m, n, k,
            alpha,
            bptr, B.ld,
            aptr, A.ld,
            beta,
            &cptr, C.ld);
    }

    // wrapper to rearrange arguments for cuBLAS
    // cublas does: B = op(A) \ B or B = B / op(A),
    // where op may be transpose or no-transpose.
    // For each config, we change as the following table:
    /*   row major  | col major
            ------------+------------
            B = A \ B   |  B' = B' / A'
            B = B / A   |  B' = A' \ B'
            B = B / A'  |  B' = A \ B'
            B = A' \ B  |  B' = B' / A' 
        */
    // Rule: switch sides A,B
    // uplo: switch 'L' to 'U' and vice versa
    template <typename T>
    void trsm(Stream* stream,
        char side, char uplo,
        bool transa, T alpha,
        Matrix<const T>& A, Matrix<T>& B)
    {
        auto aptr = AsDeviceMemory(A.data());
        auto bptr = AsDeviceMemory(B.data());

        perftools::gputools::blas::Transpose real_transa = transa ? perftools::gputools::blas::Transpose::kTranspose : perftools::gputools::blas::Transpose::kNoTranspose;

        perftools::gputools::blas::Side real_side;
        switch (side) {
        case 'L':
            real_side = perftools::gputools::blas::Side::kRight;
            break;
        case 'R':
            real_side = perftools::gputools::blas::Side::kLeft;
            break;
        }
        perftools::gputools::blas::UpperLower real_uplo;
        switch (uplo) {
        case 'U':
            real_uplo = perftools::gputools::blas::UpperLower::kLower;
            break;
        case 'L':
            real_uplo = perftools::gputools::blas::UpperLower::kUpper;
            break;
        }
        perftools::gputools::blas::Diagonal diagonal = perftools::gputools::blas::Diagonal::kNonUnit;

        stream->ThenBlasTrsm(
            real_side,
            real_uplo,
            real_transa,
            diagonal,
            B.n, B.m, alpha,
            aptr, A.ld,
            &bptr, B.ld);
    }
    template <typename T>
    bool nz(Matrix<T>& a)
    {
        bool ans = true;
        ans &= a.m > 0;
        ans &= a.n > 0;
        return ans;
    }
} //anonymous namespace
namespace functors {

    template <typename T>
    struct ComputeCholGrad<GPUDevice, T> {
        // TODO: Make this a parameter
        const int blocksize = 256;
        using Helper = CUDAMatrixHelper<T>;
        void operator()(OpKernelContext* ctx, const Tensor& Ltensor, const Tensor& Ltensorbar, Tensor* Atensorbar)
        {
            // const Eigen::GpuDevice dev = ctx->eigen_device<GPUDevice>();
            // errors with incomplete type: why??? SO SAD.
            // workaround seems to be to pass this by reference to an NVCC compiled function

            auto* stream = ctx->op_device_context()->stream();
            const T* Lptr = Ltensor.flat<T>().data();
            const T* Lbarptr = Ltensorbar.flat<T>().data();
            T* Abarptr = Atensorbar->flat<T>().data();
            int M = Ltensor.dim_size(0);
            Matrix<const T> L{ Lptr, 0, M, M, M };
            Matrix<const T> Lbar{ Lbarptr, 0, M, M, M };
            Matrix<T> Abar{ Abarptr, 0, M, M, M };

            // Copy Lbar into Abar on our stream
            Helper::copy(ctx->eigen_device<GPUDevice>(), Abar, Lbar);

            // Allocate scratch space (blocksize)
            Tensor scratchtensor;
            TensorShape tmp_shape({ blocksize, blocksize });
            ctx->allocate_temp(DataTypeToEnum<T>::value, tmp_shape, &scratchtensor);
            T* scratchptr = scratchtensor.flat<T>().data();

            T one = 1.0;
            T zero = 0.0;
            for (int k = M; k > 0; k -= blocksize) {
                int j = std::max(0, k - blocksize);
                L3Par<const T> par{ L, j, k };
                L3Par<T> parbar{ Abar, j, k };
                Matrix<const T> cAbar{ Abar.dataptr, 0, Abar.m, Abar.n, Abar.ld };
                L3Par<const T> cparbar{ cAbar, j, k };

                if (nz(par.D) && nz(parbar.C)) {
                    // std::cout << "solve_tri\n";
                    trsm(stream, 'R', 'L', false, one, par.D, parbar.C);
                }
                if (nz(parbar.C) && nz(par.R) && nz(parbar.B)) {
                    // Bbar <- Bbar - Cbar * R
                    gemm(stream, false, false, -one, cparbar.C, par.R, one, parbar.B);
                }
                if (nz(parbar.D) && nz(parbar.C) && nz(par.C)) {
                    // these next two lines same as Dbar <- Dbar - tril(Cbar' * C)
                    // Dbar <- Dbar - Cbar' * C
                    gemm(stream, true, false, -one, cparbar.C, par.C, one, parbar.D);
                    // tril(parbar.D);
                    Helper::tril(ctx->eigen_device<GPUDevice>(), parbar.D);
                }

                
                // CholeskyGradSymbolic expects (L, Lbar, P) and stores results in P
                // Therefore call as: CholeskyGradSymbolic(D, DbarCopy, Dbar);
                int dsize = k - j;
                if (nz(par.D)) {
                    // Copy Dbar into pre-allocated scratch memory.
                    Matrix<T> DbarCopy{ scratchptr, 0, dsize, dsize, dsize };
                    Helper::copy(ctx->eigen_device<GPUDevice>(), DbarCopy, cparbar.D);
                    Matrix<const T> cDbarCopy{ scratchptr, 0, dsize, dsize, dsize };
                    CholeskyGradSymbolic(ctx, par.D, cDbarCopy, parbar.D);
                }

                if (nz(parbar.C) && nz(par.B) && nz(parbar.R)) {
                    // Rbar <- Rbar - Cbar' * B
                    // std::cout << "Cbar' * B\n";
                    // gemm(-1.0f, parbar.C, CUBLAS_OP_T, par.B, CUBLAS_OP_N, 1.0f, parbar.R);
                    gemm(stream, true, false, -one, cparbar.C, par.B, one, parbar.R);
                }
                if (nz(parbar.D) && nz(par.R) && nz(parbar.R)) {
                    // Rbar <- Rbar - Dbar * R
                    gemm(stream, false, false, -one, cparbar.D, par.R, one, parbar.R);
                    // Rbar <- Rbar - Dbar' * R
                    gemm(stream, true, false, -one, cparbar.D, par.R, one, parbar.R);
                }
            }
            // Convert to the redundant array of sensitivites to match CPU version
            Helper::reflect_half(ctx->eigen_device<GPUDevice>(), Abar);
            ctx->op_device_context()->stream()->BlockHostUntilDone();
        }
        void CholeskyGradSymbolic(OpKernelContext* ctx, Matrix<const T>& L, Matrix<const T>& Lbar, Matrix<T>& Abar)
        {
            auto* stream = ctx->op_device_context()->stream();

            T one = 1.0;
            T zero = 0.0;
            // Abar <- L^T Lbar
            gemm(stream, true, false, one, L, Lbar, zero, Abar);
            // P <- Phi(L^T Lbar) + Phi(L^T Lbar)^T
            Helper::symmetrise(ctx->eigen_device<GPUDevice>(), Abar);
            // P <- L^-T(Phi(L^T Lbar) + Phi(L^T Lbar)^T)
            trsm(stream, 'L', 'L', true, one, L, Abar);
            // P <- L^-T(Phi(L^T Lbar) + Phi(L^T Lbar)^T) L^-1
            trsm(stream, 'R', 'L', false, one, L, Abar);
            // P <- Phi(L^-T(Phi(L^T Lbar) + Phi(L^T Lbar)^T) L^-1)
            Helper::phi(ctx->eigen_device<GPUDevice>(), Abar);
        }
    };
} // namespace functors
#endif // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
    CholeskyGrad<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
        .Device(DEVICE_CPU)
        .TypeConstraint<double>("T"),
    CholeskyGrad<CPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    CholeskyGrad<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    CholeskyGrad<GPUDevice, double>);

} // namespace tensorflow
