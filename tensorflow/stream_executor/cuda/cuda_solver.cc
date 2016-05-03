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

#include "tensorflow/stream_executor/cuda/cuda_solver.h"

#include <dlfcn.h>

#include <complex>

#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuSolverPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(__name)                            \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCusolverDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static void *f = dlsym(GetDsoHandle(), kName);                        \
      CHECK(f != nullptr) << "could not find " << kName                     \
                          << " in cuSolver DSO; dlerror: " << dlerror();      \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    template <typename... Args>                                             \
    cusolverStatus_t operator()(CUDAExecutor * parent, Args... args) {        \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(__name)

#define CUSOLVER_SOLVER_ROUTINE_EACH(__macro) \
  __macro(cusolverDnDpotrf)               \
  __macro(cusolverDnSpotrf)

PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(cusolverDnCreate)
PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(cusolverDnDestroy)
PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(cusolverDnSetStream)
PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP(cusolverDnGetStream)
CUSOLVER_SOLVER_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUSOLVER_WRAP)

}  // namespace dynload

static string ToString(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
      return port::StrCat("<invalid cusolver status: ", status, ">");
  }
}

bool CUDASolver::Init() {
  cusolverStatus_t ret = dynload::cusolverDnCreate(parent_, &solver_);
  if (ret != CUSOLVER_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cusolver handle: " << ToString(ret);
    return false;
  }

  return true;
}

CUDASolver::CUDASolver(cuda::CUDAExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), solver_(nullptr) {}

CUDASolver::~CUDASolver() {
  if (solver_ != nullptr) {
    dynload::cusolverDnDestroy(parent_, solver_);
  }
}

bool CUDASolver::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsCUDAStreamValue(stream) != nullptr);
  CHECK(solver_ != nullptr);
  cusolverStatus_t ret =
      dynload::cusolverDnSetStream(parent_, solver_, AsCUDAStreamValue(stream));
  if (ret != CUSOLVER_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuSolverDncalls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming solver arguments into cuSolverDn arguments.

cublasOperation_t CUDASolverTranspose(solver::Transpose trans) {
  switch (trans) {
    case solver::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case solver::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case solver::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of solver::Transpose.";
  }
}

cublasFillMode_t CUDASolverUpperLower(solver::UpperLower uplo) {
  switch (uplo) {
    case solver::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case solver::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of solver::UpperLower.";
  }
}

cublasDiagType_t CUDASolverDiagonal(solver::Diagonal diag) {
  switch (diag) {
    case solver::Diagonal::kUnit:
      return CUBLAS_DIAG_UNIT;
    case solver::Diagonal::kNonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of solver::Diagonal.";
  }
}

cublasSideMode_t CUDASolverSide(solver::Side side) {
  switch (side) {
    case solver::Side::kLeft:
      return CUBLAS_SIDE_LEFT;
    case solver::Side::kRight:
      return CUBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of solver::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool CUDASolver::DoSolverInternal(FuncT cusolver_func, Stream *stream,
                              Args... args) {
  mutex_lock lock{mu_};

  CHECK(solver_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  cusolverDnStatus_t ret = cusolver_func(parent_, solver_, args...);
  if (ret != CUSOLVER_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cuSolver routine " << cusolver_func.kName << ": "
               << ToString(ret);
    return false;
  }

  return true;
}

bool CUDASolver::DoSolverPotrf(Stream *stream, uint64 elem_count, solver::UpperLower uplo,
                          DeviceMemory<float> *A, uint64t lda,
                          ScratchAllocator *scratch_allocator) {

  int Lwork;
  if (!DoSolverInternal(dynload::cusolverDnSpotrf_bufferSize, stream,
                      elem_count, CudaSolverUpperLower(uplo),
                      CUDAMemoryMutable(A), lda, 
                      &Lwork)) {
    return false;
  }
  DeviceMemory<float> scratch;
  std::unique_ptr<TemporaryDeviceMemory<float>> scratch_temporary;
  if (scratch_allocator != nullptr) {
    scratch = scratch_allocator->AllocateBytes(stream, sizeof(float) * Lwork).ValueOrDie();
  } else {
    SE_ASSIGN_OR_RETURN(scratch_temporary,
                        stream->AllocateTemporaryArray<float>(Lwork));
    scratch = DeviceMemory<float>(*scratch_temporary->mutable_device_memory());
  }
  return DoSolverInternal(dynload::cusolverDnSpotrf, stream,
                        elem_count, CudaSolverUpperLower(uplo),
                        CUDAMemoryMutable(A), lda,
                        Lwork, CUDAMemoryMutable(scratch));
} 
bool CUDASolver::DoSolverPotrf(Stream *stream, uint64 elem_count, solver::UpperLower uplo,
                          DeviceMemory<double> *A, uint64 lda,
                          ScratchAllocator *scratch_allocator) {
  int Lwork;
  if (!DoSolverInternal(dynload::cusolverDnDpotrf_bufferSize, stream,
                      elem_count, CudaSolverUpperLower(uplo),
                      CUDAMemoryMutable(A), lda, 
                      &Lwork)) {
    return false;
  }
  DeviceMemory<double> scratch;
  if (scratch_allocator != nullptr) {
    scratch = scratch_allocator->AllocateBytes(stream, sizeof(double) * Lwork).ValueOrDie();
  } else {
    SE_ASSIGN_OR_RETURN(scratch_temporary,
                        stream->AllocateTemporaryArray<double>(Lwork));
    scratch = DeviceMemory<double>(*scratch_temporary->mutable_device_memory());
  }
  return DoSolverInternal(dynload::cusolverDnDpotrf, stream,
                        elem_count, CudaSolverUpperLower(uplo),
                        CUDAMemoryMutable(A), lda,
                        Lwork, CUDAMemoryMutable(scratch));
}

}  // namespace cuda

namespace gpu = ::perftools::gputools;

void initialize_cusolver() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::SolverFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuSolverPlugin, "cuSolverDn",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::solver::SolverSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuSolver "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDASolver *solver =
                    new gpu::cuda::CUDASolver(cuda_executor);
                if (!solver_->Init()) {
                  // Note: Init() will log a more specific error.
                  delete solver;
                  return nullptr;
                }
                return solver;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuSolverDn factory: "
               << status.error_message();
  }

  // Prime the cuSOLVER DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCusolverDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuSolver DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kSolver,
                                                     gpu::cuda::kCuSolverPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_cusolver,
                            { perftools::gputools::initialize_cusolver(); });
