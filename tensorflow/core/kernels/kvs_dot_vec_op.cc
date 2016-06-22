// #define EIGEN_USE_GPU

#include <memory>
#include <vector>
#include "tensorflow/core/framework/op.h"

#include "tensorflow/core/kernels/kvs_dot_vec_op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#include "kvs_dot_vec_op.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/lib/status_macros.h"

#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/core/kernels/transpose_functor.h"

// #include "tensorflow/tensorflow/core/kernels/"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Type>
using DeviceMemory = perftools::gputools::DeviceMemory<Type>;

template <typename Type>
using TemporaryDeviceMemory = perftools::gputools::TemporaryDeviceMemory<Type>;


template <typename T, int D>
struct kvs_dot_vec_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
                    const Tensor* vec, Tensor* product) {
        const T * kvs_t_ptrs[D];

        // allocate D transpose tensors and transpose into them
        std::vector<Tensor> tmp_tensors;
        for (int d = 0; d < D; ++d) {
            tmp_tensors.push_back(Tensor{});
        }
        for (int d = 0; d < D; ++d) {
            TensorShape old_shape = kvs[d].shape();
            TensorShape new_shape{old_shape.dim_size(1), old_shape.dim_size(0)};
            Tensor& tmp = tmp_tensors[d];
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, new_shape,
                       &tmp));
            OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_device<GPUDevice>(), 
                kvs[d],
                {1,0}, &tmp));
            kvs_t_ptrs[d] = tmp.flat<T>().data();
        }



        // copy pointers to kvs to device memory
        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_t_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_t_dptrs, &kvs_t_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int H = kvs[0].dim_size(0);
        int wcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            wcols[d] = kvs[d].dim_size(1);
            vec_len *= kvs[d].dim_size(1);
        }
        int64_t wprods[D];
        int64_t wprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            wprods_tmp /= wcols[d];
            wprods[d] = wprods_tmp;
        }

        // allocate and copy wcols, wprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_wcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_wprods_tmp;
        d_wcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_wprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_wcols = 
            DeviceMemory<int>(*d_wcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_wprods = 
            DeviceMemory<int64_t>(*d_wprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_wcols, &wcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_wprods, &wprods[0], sizeof(int64_t)*D);


        launch_cu_kvs_dot_vec<T, D>(H, 
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wprods),
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_t_dptrs), 
            vec->flat<T>().data(), product->flat<T>().data()
        );
    }
};

template <typename T, int D>
struct kvs_dot_vec_kvsgrad_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, const OpInputList& kvs, 
                    const Tensor* vec, const Tensor* ingrad,
                    OpOutputList* kvsgrad) {
        const T * kvs_t_ptrs[D];
        const T * grad_ptrs[D];

        // allocate D transpose tensors and transpose into them
        std::vector<Tensor> tmp_tensors;
        std::vector<Tensor> grad_tensors;
        for (int d = 0; d < D; ++d) {
            tmp_tensors.push_back(Tensor{});
            grad_tensors.push_back(Tensor{});
        }
        for (int d = 0; d < D; ++d) {
            TensorShape old_shape = kvs[d].shape();
            TensorShape new_shape{old_shape.dim_size(1), old_shape.dim_size(0)};
            Tensor& tmp = tmp_tensors[d];
            Tensor& grad = grad_tensors[d];
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, new_shape,
                       &tmp));
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, new_shape,
                       &grad));
            OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_device<GPUDevice>(), 
                kvs[d],
                {1,0}, &tmp));
            kvs_t_ptrs[d] = tmp.flat<T>().data();
            grad_ptrs[d] = grad.flat<T>().data();
        }



        // copy pointers to kvs and grad to device memory
        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        std::unique_ptr<TemporaryDeviceMemory<T *>> grad_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        grad_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_t_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        DeviceMemory<T*> grad_dptrs = 
            DeviceMemory<T*>(*grad_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_t_dptrs, &kvs_t_ptrs[0], sizeof(T*)*D);
        stream->ThenMemcpy(&grad_dptrs, &grad_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int H = kvs[0].dim_size(0);
        int wcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            wcols[d] = kvs[d].dim_size(1);
            vec_len *= kvs[d].dim_size(1);
        }
        int64_t wprods[D];
        int64_t wprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            wprods_tmp /= wcols[d];
            wprods[d] = wprods_tmp;
        }

        // allocate and copy wcols, wprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_wcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_wprods_tmp;
        d_wcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_wprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_wcols = 
            DeviceMemory<int>(*d_wcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_wprods = 
            DeviceMemory<int64_t>(*d_wprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_wcols, &wcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_wprods, &wprods[0], sizeof(int64_t)*D);


        launch_cu_kvs_dot_vec_kvsgrad<T, D>(H, 
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wprods),
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_t_dptrs), 
            vec->flat<T>().data(), ingrad->flat<T>().data(),
            perftools::gputools::cuda::CUDAMemoryMutable(&grad_dptrs)
        );

        for (int d = 0; d < D; ++d) {
            tmp_tensors[d] = Tensor{}; // release reference to temp tensor
            TensorShape outshape = kvs[d].shape();
            Tensor * out = nullptr;
            OP_REQUIRES_OK(ctx, kvsgrad->allocate(d, outshape, &out));
            OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_device<GPUDevice>(), 
                grad_tensors[d],
                {1,0}, out));
        }

    }
};


template <typename T, int D>
struct kvs_dot_vec_vecgrad_functor<GPUDevice, T, D> {
    void operator()(OpKernelContext* ctx, 
                    const OpInputList& kvs, 
                    const Tensor* ingrad,
                    Tensor* vecgrad) {
        const T * kvs_t_ptrs[D];

        // allocate D transpose tensors and transpose into them
        std::vector<Tensor> tmp_tensors;
        for (int d = 0; d < D; ++d) {
            tmp_tensors.push_back(Tensor{});
        }
        for (int d = 0; d < D; ++d) {
            TensorShape old_shape = kvs[d].shape();
            TensorShape new_shape{old_shape.dim_size(1), old_shape.dim_size(0)};
            Tensor& tmp = tmp_tensors[d];
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, new_shape,
                       &tmp));
            OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_device<GPUDevice>(), 
                kvs[d],
                {1,0}, &tmp));
            kvs_t_ptrs[d] = tmp.flat<T>().data();
        }



        // copy pointers to kvs to device memory
        auto* stream = ctx->op_device_context()->stream();
        // allocate temporary memory to store our pointers-pointers
        std::unique_ptr<TemporaryDeviceMemory<T *>> kvs_dptrs_tmp;
        kvs_dptrs_tmp = stream->AllocateTemporaryArray<T *>(D).ConsumeValueOrDie();
        DeviceMemory<T*> kvs_t_dptrs = 
            DeviceMemory<T*>(*kvs_dptrs_tmp->mutable_device_memory());
        // copy host-side pointers to kvs to device
        stream->ThenMemcpy(&kvs_t_dptrs, &kvs_t_ptrs[0], sizeof(T*)*D);

        // Prepare shape data for the GPU
        int H = kvs[0].dim_size(0);
        int wcols[D];
        int64_t vec_len = 1;
        for (int d = 0; d < D; ++d) {
            wcols[d] = kvs[d].dim_size(1);
            vec_len *= kvs[d].dim_size(1);
        }
        int64_t wprods[D];
        int64_t wprods_tmp = vec_len;
        for (int d = 0; d < D; ++d) {
            wprods_tmp /= wcols[d];
            wprods[d] = wprods_tmp;
        }

        // allocate and copy wcols, wprods to GPU
        std::unique_ptr<TemporaryDeviceMemory<int>> d_wcols_tmp;
        std::unique_ptr<TemporaryDeviceMemory<int64_t>> d_wprods_tmp;
        d_wcols_tmp = stream->AllocateTemporaryArray<int>(D).ConsumeValueOrDie();
        d_wprods_tmp = stream->AllocateTemporaryArray<int64_t>(D).ConsumeValueOrDie();
        DeviceMemory<int> d_wcols = 
            DeviceMemory<int>(*d_wcols_tmp->mutable_device_memory());
        DeviceMemory<int64_t> d_wprods = 
            DeviceMemory<int64_t>(*d_wprods_tmp->mutable_device_memory());
        stream->ThenMemcpy(&d_wcols, &wcols[0], sizeof(int)*D);
        stream->ThenMemcpy(&d_wprods, &wprods[0], sizeof(int64_t)*D);


        launch_cu_kvs_dot_vec_vecgrad<T, D>(H, vec_len,
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wcols),
            perftools::gputools::cuda::CUDAMemoryMutable(&d_wprods),
            perftools::gputools::cuda::CUDAMemoryMutable(&kvs_t_dptrs), 
            ingrad->flat<T>().data(),
            vecgrad->flat<T>().data()
        );
    }
};



template <typename Device, typename T>
class KvsDotVec : public OpKernel {
    public:
	    explicit KvsDotVec(OpKernelConstruction* context) : OpKernel(context) {};

	    void Compute(OpKernelContext* ctx) override {
	    
	        OpInputList kvs_list;
    		OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * vec = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("vec", &vec));
            // std::cout << vec->dim_size(0) << " " << vec->dim_size(1) << std::endl;
    		const int N = kvs_list.size();
    		// std::cout << N << std::endl;

            Tensor * out = nullptr;
            TensorShape outshape;
            outshape.AddDim(kvs_list[0].dim_size(0));
            outshape.AddDim(1);

            OP_REQUIRES_OK(ctx, ctx->allocate_output
                (0, outshape, &out));
            switch (N) {
                case 2:
                    kvs_dot_vec_functor<GPUDevice, T, 2> k2;
                    k2(ctx, kvs_list, vec, out);
                    break;
                case 3:
                    kvs_dot_vec_functor<GPUDevice, T, 3> k3;
                    k3(ctx, kvs_list, vec, out);
                    break;
                case 4:
                    kvs_dot_vec_functor<GPUDevice, T, 4> k4;
                    k4(ctx, kvs_list, vec, out);
                    break;
                case 5:
                    kvs_dot_vec_functor<GPUDevice, T, 5> k5;
                    k5(ctx, kvs_list, vec, out);
                    break;
                case 6:
                    kvs_dot_vec_functor<GPUDevice, T, 6> k6;
                    k6(ctx, kvs_list, vec, out);
                    break;
                case 7:
                    kvs_dot_vec_functor<GPUDevice, T, 7> k7;
                    k7(ctx, kvs_list, vec, out);
                    break;
                case 8:
                    kvs_dot_vec_functor<GPUDevice, T, 8> k8;
                    k8(ctx, kvs_list, vec, out);
                    break;
                case 9:
                    kvs_dot_vec_functor<GPUDevice, T, 9> k9;
                    k9(ctx, kvs_list, vec, out);
                    break;
                case 10:
                    kvs_dot_vec_functor<GPUDevice, T, 10> k10;
                    k10(ctx, kvs_list, vec, out);
                    break;
            }
	    }
};

template <typename Device, typename T>
class KvsDotVecKvsGrad : public OpKernel {
    public:
        explicit KvsDotVecKvsGrad(OpKernelConstruction* context) : OpKernel(context) {};

        void Compute(OpKernelContext* ctx) override {
        
            OpInputList kvs_list;
            OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * vec = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("vec", &vec));
            const Tensor * ingrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("ingrad", &ingrad));
            const int N = kvs_list.size();

            OpOutputList kvsgrad;
            OP_REQUIRES_OK(ctx, ctx->output_list("kvsgrad", &kvsgrad));

            switch (N) {
                case 2:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 2> k2;
                    k2(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 3:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 3> k3;
                    k3(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 4:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 4> k4;
                    k4(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 5:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 5> k5;
                    k5(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 6:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 6> k6;
                    k6(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 7:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 7> k7;
                    k7(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 8:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 8> k8;
                    k8(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 9:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 9> k9;
                    k9(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
                case 10:
                    kvs_dot_vec_kvsgrad_functor<GPUDevice, T, 10> k10;
                    k10(ctx, kvs_list, vec, ingrad,
                        &kvsgrad);
                    break;
            }
        }
};

template <typename Device, typename T>
class KvsDotVecVecGrad : public OpKernel {
    public:
        explicit KvsDotVecVecGrad(OpKernelConstruction* context) : OpKernel(context) {};

        void Compute(OpKernelContext* ctx) override {
        
            OpInputList kvs_list;
            OP_REQUIRES_OK(ctx, ctx->input_list("kvs", &kvs_list));
            const Tensor * ingrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("ingrad", &ingrad));
            
            const int N = kvs_list.size();
            int64_t vec_len = 1;
            for (int d = 0; d < N; ++d) {
                vec_len *= kvs_list[d].dim_size(1);
            }  
            TensorShape vecgrad_shape{vec_len, 1};
            Tensor * vecgrad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("vecgrad", vecgrad_shape, &vecgrad));

            switch (N) {
                case 2:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 2> k2;
                    k2(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 3:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 3> k3;
                    k3(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 4:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 4> k4;
                    k4(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 5:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 5> k5;
                    k5(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 6:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 6> k6;
                    k6(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 7:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 7> k7;
                    k7(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 8:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 8> k8;
                    k8(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 9:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 9> k9;
                    k9(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
                case 10:
                    kvs_dot_vec_vecgrad_functor<GPUDevice, T, 10> k10;
                    k10(ctx, kvs_list, ingrad,
                        vecgrad);
                    break;
            }
        }
};


REGISTER_KERNEL_BUILDER(
    Name("KvsDotVec")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    KvsDotVec<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("KvsDotVec")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    KvsDotVec<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("KvsDotVecKvsGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    KvsDotVecKvsGrad<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("KvsDotVecKvsGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    KvsDotVecKvsGrad<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("KvsDotVecVecGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    KvsDotVecVecGrad<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("KvsDotVecVecGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
    KvsDotVecVecGrad<GPUDevice, double>);
}