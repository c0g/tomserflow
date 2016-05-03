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

// Exposes the family of Solver routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsSolver() for details.
//
// This abstraction makes it simple to entrain Solver operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<float> x = stream_exec->AllocateArray<float>(1024);
//  DeviceMemory<float> y = stream_exec->AllocateArray<float>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  stream
//    .Init()
//    .ThenSolverPotrf(1024*1024, kLower, x, 1024)
//    .BlockHostUntilDone();
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned Solver
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOR_SOLVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_SOLVER_H_

#include <complex>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/lib/array_slice.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class Stream;
class ScratchAllocator;

template <typename ElemT>
class DeviceMemory;

namespace solver {

// Specifies whether the input matrix will be transposed or
// transposed+conjugated before any SOLVER operations.
enum class Transpose { kNoTranspose, kTranspose, kConjugateTranspose };

// Returns a name for t.
string TransposeString(Transpose t);

// Specifies whether the upper or lower triangular part of a
// symmetric/Hermitian matrix is used.
enum class UpperLower { kUpper, kLower };

// Returns a name for ul.
string UpperLowerString(UpperLower ul);

// Specifies whether a matrix is unit triangular.
enum class Diagonal { kUnit, kNonUnit };

// Returns a name for d.
string DiagonalString(Diagonal d);

// Specifies whether a Hermitian matrix appears on the left or right in
// operation.
enum class Side { kLeft, kRight };

// Returns a name for s.
string SideString(Side s);

// SOLVER support interface -- this can be derived from a GPU executor when the
// underlying platform has an SOLVER library implementation available. See
// StreamExecutor::AsSolver().
//
// Thread-hostile: CUDA associates a CUDA-context with a particular thread in
// the system. Any operation that a user attempts to perform by enqueueing SOLVER
// operations on a thread not-associated with the CUDA-context has unknown
// behavior at the current time; see b/13176597
class SolverSupport {
 public:
  virtual ~SolverSupport() {}

  /*  DPOTRF computes the Cholesky factorization of a real symmetric */
  /*  positive definite matrix A. */

  /*  The factorization has the form */
  /*     A = U**T * U,  if UPLO = 'U', or */
  /*     A = L  * L**T,  if UPLO = 'L', */
  /*  where U is an upper triangular matrix and L is lower triangular. */
  virtual bool DoSolverPotrf(Stream *stream, uint64 elem_count, UpperLower uplo,
                          DeviceMemory<float> *A, uint64 lda,
                          ScratchAllocator* scratch_allocator) = 0;


 protected:
  SolverSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(SolverSupport);
};

// Macro used to quickly declare overrides for abstract virtuals in the
// SolverSupport base class.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_SOLVER_SUPPORT_OVERRIDES                \
  bool DoSolverPotrf(Stream *stream, uint64 elem_count, UpperLower uplo,       \
                          DeviceMemory<float> *A, uint64 lda,                  \
                          ScratchAllocator* scratch_allocator) override;

}  // namespace solver
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_SOLVER_H_
