// #pragma once
// #include "tensorflow/core/framework/tensor_types.h"
// #include "tensorflow/core/user_ops/gpu_cholgrad_func.h"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/framework/op_kernel.h"

// using CPUDevice = Eigen::ThreadPoolDevice;

// namespace tensorflow {
// namespace functors {
//     template <typename T>
//     struct ComputeCholGrad<CPUDevice, T> {
//         using Matrix =
//           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//         using ConstMatrixMap = Eigen::Map<const Matrix>;
//         using MatrixMap = Eigen::Map<Matrix>;
//         using ConstRef = Eigen::Ref<const Matrix>;
//         using Ref = Eigen::Ref<Matrix>;
//         void operator ()(OpKernelContext* ctx, const Tensor& Ltensor, const Tensor& Lbartensor, Tensor* Abartensor)
//         {
//             // take Eigen matrix view into the tensors
//             int M = Ltensor.dim_size(0);
//             const ConstMatrixMap Lfull(Ltensor.flat<T>().data(), M, M);
//             const ConstMatrixMap Lbar(Lbartensor.flat<T>().data(), M, M);
//             MatrixMap Abar(Abartensor->flat<T>().data(), M, M);

//             // Algorithm only depends on lower triangular half on Ltensor.
//             const Matrix L = Lfull.template triangularView<Eigen::Lower>();
//             // Algorithm only depends on lower triangular half on Lbartensor.
//             Abar = Lbar.template triangularView<Eigen::Lower>();

//             const int64 kMatrixSize = L.rows();
//             const int64 kMaxBlockSize = 32;

//             for (int64 block_end = kMatrixSize; block_end > 0;
//                  block_end -= kMaxBlockSize) {
//                 /* 
//                   This shows the block structure.

//                   /      \
//                   |      |
//                   | R D  |
//                   \ B C  /

//                   Variables names representing the derivative matrix have a trailing '_bar'.
//                 */

//                 const int64 block_begin = std::max(0ll, block_end - kMaxBlockSize);
//                 const int64 block_size = block_end - block_begin;
//                 const int64 trailing_size = kMatrixSize - block_end;

//                 auto B = L.block(block_end, 0, trailing_size, block_begin);
//                 auto B_bar = Abar.block(block_end, 0, trailing_size, block_begin);

//                 auto C = L.block(block_end, block_begin, trailing_size,
//                     block_size);
//                 auto C_bar = Abar.block(block_end, block_begin, trailing_size,
//                     block_size);

//                 auto D = L.block(block_begin, block_begin, block_size,
//                     block_size);
//                 auto D_bar = Abar.block(block_begin, block_begin, block_size, block_size);

//                 auto R = L.block(block_begin, 0, block_size, block_begin);
//                 auto R_bar = Abar.block(block_begin, 0, block_size, block_begin);

//                 C_bar = D.adjoint().template triangularView<Eigen::Upper>().solve(C_bar.adjoint()).adjoint();
//                 D_bar -= (C_bar.adjoint() * C).template triangularView<Eigen::Lower>();
//                 B_bar -= C_bar * R;
//                 R_bar -= C_bar.adjoint() * B;
//                 CholeskyGradUnblocked(D, D_bar);
//                 R_bar -= (D_bar + D_bar.adjoint()) * R;
//             }
//             Abar = (0.5 * (Abar + Abar.transpose())).eval();
//         }
//         void CholeskyGradUnblocked(const ConstRef l_block, Ref grad_block)
//         {
//             const int64 kMatrixSize = l_block.rows();
//             for (int64 k = kMatrixSize - 1; k >= 0; k--) {
//                 /* This shows the block structure.

//                   /      \
//                   |      |
//                   | r d  |
//                   \ B c  /

//                   Variables names representing the derivative matrix have a trailing '_bar'.
//                   */

//                 const int64 number_rows_B = kMatrixSize - (k + 1);
//                 const int64 number_rows_r_stack_B = number_rows_B + 1;

//                 auto r = l_block.block(k, 0, 1, k);
//                 auto r_bar = grad_block.block(k, 0, 1, k);
//                 auto d = l_block(k, k); // This needs to be a scalar rather than a view.
//                 auto d_bar = grad_block.block(k, k, 1, 1);
//                 // B is not included explicitly because it is not used on its own.
//                 auto B_bar = grad_block.block(k + 1, 0, number_rows_B, k);
//                 auto c = l_block.block(k + 1, k, number_rows_B, 1);
//                 auto c_bar = grad_block.block(k + 1, k, number_rows_B, 1);
//                 // Result of vertical stacking d_bar and c_bar.
//                 auto d_stack_c_bar = grad_block.block(k, k, number_rows_r_stack_B, 1);
//                 // Result of vertical stacking of r and B.
//                 auto r_stack_B = l_block.block(k, 0, number_rows_r_stack_B, k);
//                 d_bar -= (c.adjoint() * c_bar) / d;
//                 d_stack_c_bar /= d;
//                 r_bar -= d_stack_c_bar.adjoint() * r_stack_B;
//                 B_bar -= c_bar * r;
//                 d_bar /= 2.;
//             }
//         }
//     };
// } // namespace functors
// } // namespace tensorflow