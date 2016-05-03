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

// CUDA-specific support for SOLVER functionality -- this wraps the cuSolverDn library
// capabilities, and is only included into CUDA implementation code -- it will
// not introduce cuda headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_H_

#include "tensorflow/stream_executor/solver.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"

typedef struct cuSolverDnContext *cusolverDnHandle_t;

namespace perftools {
namespace gputools {

class Stream;

namespace cuda {

// Opaque and unique identifier for the cuSolverDn plugin.
extern const PluginId kCuSolverDNPlugin;

class CUDAExecutor;

// Solver plugin for CUDA platform via cuSolverDn library.
//
// This satisfies the platform-agnostic SolverSupport interface.
//
// Note that the cuSolverDn handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent CUDAExecutor is tied
// to. This simply happens as an artifact of creating the cuSolver handle when a
// CUDA context is active.
//
// Thread-safe post-initialization.
class CUDASolver : public solver::SolverSupport {
 public:
  explicit CUDASolver(CUDAExecutor *parent);

  // Allocates a cuSolver handle.
  bool Init();

  // Releases the cuSolver handle, if present.
  ~CUDASolver() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_SOLVER_SUPPORT_OVERRIDES

 private:
  // Tells cuSolver to enqueue the Solver operation onto a particular Stream.
  //
  // cuSolver is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into cuSolver.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real cuSolver function together with error
  // handling.
  //
  // cusolver_func:        cuSolverDn function pointer.
  // cusolver_name:        cuSolverDn function name.
  // stream:             Stream to enqueue the Solver operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // args:               Arguments of cuSolverDn function.
  template <typename FuncT, typename... Args>
  bool DoSolverInternal(FuncT cusolver_func, Stream *stream, bool pointer_mode_host,
                      Args... args);


  // mutex that guards the cuSolverDn handle for this device.
  mutex mu_;

  // CUDAExecutor which instantiated this CUDASolver.
  // Immutable post-initialization.
  CUDAExecutor *parent_;

  // cuSolverDn library handle on the device.
  cusolverDnHandle_t solver_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(CUDASolver);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_H_
