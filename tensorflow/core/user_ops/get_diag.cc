
//Copyright (c) 2016, GPU code Tom Nickson. All rights reserved.
//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "get_diag.h"


REGISTER_OP("GetDiag").Input("l: T").Output("g: T").Attr("T : {float, double}").Doc("Get diagonal of a square matrix.");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {
template <typename Device, typename T>
class GetDiag : public OpKernel {
    public:
    
        explicit GetDiag(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {

            const Tensor& input_tensor = context->input(0);
            
            // Ensure is a square matrix
            OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()), errors::InvalidArgument("In[0] is not a matrix"));
            OP_REQUIRES(context, input_tensor.dim_size(0) == input_tensor.dim_size(1), errors::InvalidArgument("Input matrix must be square."));        

            const int N = input_tensor.dim_size(0);

            Tensor* output_tensor = NULL;
            
            const TensorShape output_shape = TensorShape({ N }); 
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));    
            auto in = input_tensor.flat<T>().data();
            auto out = output_tensor->flat<T>().data();
            functors::get_diag<Device, T> diag;
            diag(context->eigen_device<Device>(), in, out, N);
        
        }
};

REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    GetDiag<CPUDevice, float>);
    
REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    GetDiag<CPUDevice,double>);

REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_GPU),
    GetDiag<GPUDevice, float>);
    
REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    GetDiag<GPUDevice, double>);
}//namespace tensorflow