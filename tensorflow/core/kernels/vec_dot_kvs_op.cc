// #define EIGEN_USE_GPU

#include <memory>
#include <vector>
#include "tensorflow/core/framework/op.h"

#include "tensorflow/core/kernels/vec_dot_kvs_op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/lib/status_macros.h"

#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/core/kernels/transpose_functor.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/kernels/fill_functor.h"



namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Type>
using DeviceMemory = perftools::gputools::DeviceMemory<Type>;

template <typename Type>
using TemporaryDeviceMemory = perftools::gputools::TemporaryDeviceMemory<Type>;


template <typename T, int D>
struct vec_dot_kvs_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, const Tensor* vec, 
        const OpInputList& kvs,Tensor* product) {
        
        const T * kvs_ptrs[D];
        for (int d = 0; d < D; ++d) {
            kvs_ptrs[d] = kvs[d].flat<T>().data();
        }



        // copy pointers to kvs to device memory
        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_dptrs, &kvs_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int W = kvs[0].dim_size(1);
        int hcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            hcols[d] = kvs[d].dim_size(0);
            vec_len *= kvs[d].dim_size(0);
        }
        int64_t hprods[D];
        int64_t hprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            hprods_tmp /= hcols[d];
            hprods[d] = hprods_tmp;
        }

        // allocate and copy hcols, hprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_hcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_hprods_tmp;
        d_hcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_hprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_hcols = 
            DeviceMemory<int>(*d_hcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_hprods = 
            DeviceMemory<int64_t>(*d_hprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_hcols, &hcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_hprods, &hprods[0], sizeof(int64_t)*D);


        launch_cu_vec_dot_kvs<T, D>(W, vec_len, 
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hprods),
            vec->flat<T>().data(), 
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_dptrs), 
            product->flat<T>().data()
        );
    }
};

template <typename T, int D>
struct vec_dot_kvs_kvsgrad_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, const Tensor* vec,
                    const OpInputList& kvs, const Tensor* ingrad,
                    OpOutputList* kvsgrad) {
        const T * kvs_ptrs[D];
        const T * grad_ptrs[D];

        for (int d = 0; d < D; ++d) {
            kvs_ptrs[d] = kvs[d].flat<T>().data();
            TensorShape outshape = kvs[d].shape();
            Tensor * out = nullptr;
            OP_REQUIRES_OK(ctx, kvsgrad->allocate(d, outshape, &out));
            grad_ptrs[d] = out->flat<T>().data();
        }



        // copy pointers to kvs and grad to device memory
        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        std::unique_ptr<TemporaryDeviceMemory<T *>> grad_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        grad_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        DeviceMemory<T*> grad_dptrs = 
            DeviceMemory<T*>(*grad_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_dptrs, &kvs_ptrs[0], sizeof(T*)*D);
        stream->ThenMemcpy(&grad_dptrs, &grad_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int W = kvs[0].dim_size(1);
        int hcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            hcols[d] = kvs[d].dim_size(0);
            vec_len *= kvs[d].dim_size(0);;
        }
        int64_t hprods[D];
        int64_t hprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            hprods_tmp /= hcols[d];
            hprods[d] = hprods_tmp;
        }

        // allocate and copy hcols, hprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_hcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_hprods_tmp;
        d_hcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_hprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_hcols = 
            DeviceMemory<int>(*d_hcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_hprods = 
            DeviceMemory<int64_t>(*d_hprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_hcols, &hcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_hprods, &hprods[0], sizeof(int64_t)*D);


        launch_cu_vec_dot_kvs_kvsgrad<T, D>(W, vec_len,
            hcols,
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hprods),
            vec->flat<T>().data(),
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_dptrs), 
            ingrad->flat<T>().data(),
            perftools::gputools::cuda::CUDAMemoryMutable(&grad_dptrs)
        );

    }
};


template <typename T, int D>
struct vec_dot_kvs_vecgrad_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, 
                    const OpInputList& kvs, 
                    const Tensor* ingrad,
                    Tensor* vecgrad) {

        const T * kvs_ptrs[D];
        for (int d = 0; d < D; ++d) {
            kvs_ptrs[d] = kvs[d].flat<T>().data();
        }



        // copy pointers to kvs to device memory

        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_dptrs, &kvs_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int H = kvs[0].dim_size(1);
        int hcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            hcols[d] = kvs[d].dim_size(0);
            vec_len *= kvs[d].dim_size(0);
        }
        int64_t hprods[D];
        int64_t hprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            hprods_tmp /= hcols[d];
            hprods[d] = hprods_tmp;
        }

        // allocate and copy hcols, hprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_hcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_hprods_tmp;
        d_hcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_hprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_hcols = 
            DeviceMemory<int>(*d_hcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_hprods = 
            DeviceMemory<int64_t>(*d_hprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_hcols, &hcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_hprods, &hprods[0], sizeof(int64_t)*D);


        launch_cu_vec_dot_kvs_vecgrad<T, D>(H, vec_len,
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_hprods),
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_dptrs), 
            ingrad->flat<T>().data(),
            vecgrad->flat<T>().data()
        );
    }
};



template <typename Device, typename T>
class VecDotKvs : public OpKernel {
    public:
	    explicit VecDotKvs(OpKernelConstruction* context) : OpKernel(context) {};

	    void Compute(OpKernelContext* ctx) override {

           

	    
	        OpInputList kvs_list;
    		OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * vec = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("vec", &vec));

            const int N = kvs_list.size();
            int64_t vec_len = 1;

            const int kvs0_width = kvs_list[0].dim_size(1);
            for (int d = 0; d < N; ++d) {
                vec_len *= kvs_list[d].dim_size(0);
                OP_REQUIRES(ctx, kvs0_width == kvs_list[d].dim_size(1), errors::InvalidArgument(
                "all kvs elements must have the same number of colums"));
            }

            OP_REQUIRES(ctx, vec->dim_size(0) == 1, errors::InvalidArgument(
                    "vec must be 1xN"));
            OP_REQUIRES(ctx, vec_len == vec->dim_size(1), errors::InvalidArgument(
                "product of kvs dim0 must == vec dim1"));

            Tensor * out = nullptr;
            TensorShape outshape;
            outshape.AddDim(1);
            outshape.AddDim(kvs_list[0].dim_size(1));
            OP_REQUIRES_OK(ctx, ctx->allocate_output
                (0, outshape, &out));
            functor::SetZeroFunctor<Device, T> zero;
            zero(ctx->eigen_device<Device>(), out->flat<T>());

            switch (N) {
                case 2:
                    vec_dot_kvs_functor<GPUDevice, T, 2> k2;
                    k2(ctx, vec, kvs_list, out);
                    break;
                case 3:
                    vec_dot_kvs_functor<GPUDevice, T, 3> k3;
                    k3(ctx, vec, kvs_list, out);
                    break;
                case 4:
                    vec_dot_kvs_functor<GPUDevice, T, 4> k4;
                    k4(ctx, vec, kvs_list, out);
                    break;
                case 5:
                    vec_dot_kvs_functor<GPUDevice, T, 5> k5;
                    k5(ctx, vec, kvs_list, out);
                    break;
                case 6:
                    vec_dot_kvs_functor<GPUDevice, T, 6> k6;
                    k6(ctx, vec, kvs_list, out);
                    break;
                case 7:
                    vec_dot_kvs_functor<GPUDevice, T, 7> k7;
                    k7(ctx, vec, kvs_list, out);
                    break;
                case 8:
                    vec_dot_kvs_functor<GPUDevice, T, 8> k8;
                    k8(ctx, vec, kvs_list, out);
                    break;
                case 9:
                    vec_dot_kvs_functor<GPUDevice, T, 9> k9;
                    k9(ctx, vec, kvs_list, out);
                    break;
                case 10:
                    vec_dot_kvs_functor<GPUDevice, T, 10> k10;
                    k10(ctx, vec, kvs_list, out);
                    break;
            }
	    }
};

template <typename Device, typename T>
class VecDotKvsKvsGrad : public OpKernel {
    public:
        explicit VecDotKvsKvsGrad(OpKernelConstruction* context) : OpKernel(context) {};

        void Compute(OpKernelContext* ctx) override {

        
            OpInputList kvs_list;
            OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * vec = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("vec", &vec));
            const Tensor * ingrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("ingrad", &ingrad));

            const int N = kvs_list.size();
            int64_t vec_len = 1;
            for (int d = 0; d < N; ++d) {
                vec_len *= kvs_list[d].dim_size(0);
            } 
            OP_REQUIRES(ctx, vec->dim_size(0) == 1, errors::InvalidArgument(
                    "vec must be 1xN"));
            OP_REQUIRES(ctx, vec_len == vec->dim_size(1), errors::InvalidArgument(
                "product of kvs dim0 must == vec dim1"));

            const int kvs0_width = kvs_list[0].dim_size(1);
            for (int d = 1; d < N; ++d) {
                OP_REQUIRES(ctx, kvs0_width == kvs_list[d].dim_size(1), errors::InvalidArgument(
                "all kvs elements must have the same number of colums"));
            }

            OpOutputList kvsgrad;
            OP_REQUIRES_OK(ctx, ctx->output_list("kvsgrad", &kvsgrad));

            switch (N) {
                case 2:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 2> k2;
                    k2(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 3:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 3> k3;
                    k3(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 4:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 4> k4;
                    k4(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 5:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 5> k5;
                    k5(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 6:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 6> k6;
                    k6(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 7:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 7> k7;
                    k7(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 8:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 8> k8;
                    k8(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 9:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 9> k9;
                    k9(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
                case 10:
                    vec_dot_kvs_kvsgrad_functor<GPUDevice, T, 10> k10;
                    k10(ctx, vec, kvs_list, ingrad,
                        &kvsgrad);
                    break;
            }
        }
};

template <typename Device, typename T>
class VecDotKvsVecGrad : public OpKernel {
    public:
        explicit VecDotKvsVecGrad(OpKernelConstruction* context) : OpKernel(context) {};

        void Compute(OpKernelContext* ctx) override {
        
            OpInputList kvs_list;
            OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * ingrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("ingrad", &ingrad));
            
            const int N = kvs_list.size();
            int64_t vec_len = 1;
            const int kvs0_width = kvs_list[0].dim_size(1);
            // std::cout << "vec_kvs_vecgrad is " << N << std::endl;
            for (int d = 0; d < N; ++d) {
                OP_REQUIRES(ctx, kvs0_width == kvs_list[d].dim_size(1), errors::InvalidArgument(
                "all kvs elements must have the same number of colums"));
                // std::cout << kvs_list[d].dim_size(0) << " ";
                vec_len *= kvs_list[d].dim_size(0);
            }
            // std::cout << std::endl;

            OP_REQUIRES(ctx, ingrad->dim_size(1) == kvs0_width, errors::InvalidArgument(
                "ingrad and kvs elements must have the same number of colums"));

            TensorShape vecgrad_shape{1, vec_len};
            Tensor * vecgrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("vecgrad", vecgrad_shape, &vecgrad));

            switch (N) {
                case 2:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 2> k2;
                    k2(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 3:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 3> k3;
                    k3(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 4:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 4> k4;
                    k4(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 5:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 5> k5;
                    k5(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 6:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 6> k6;
                    k6(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 7:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 7> k7;
                    k7(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 8:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 8> k8;
                    k8(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 9:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 9> k9;
                    k9(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 10:
                    vec_dot_kvs_vecgrad_functor<GPUDevice, T, 10> k10;
                    k10(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
            }
        }
};


REGISTER_KERNEL_BUILDER(
    Name("VecDotKvs")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    VecDotKvs<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("VecDotKvs")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    VecDotKvs<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("VecDotKvsKvsGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    VecDotKvsKvsGrad<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("VecDotKvsKvsGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    VecDotKvsKvsGrad<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("VecDotKvsVecGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    VecDotKvsVecGrad<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("VecDotKvsVecGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    VecDotKvsVecGrad<GPUDevice, double>);
}