//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.

#include "tensorflow/core/framework/op.h"

REGISTER_OP("Triangle").Input("a: T").Output("x: T").Attr( "T: {float, double}").Attr( "Case: {'upper','lower' }" ).Doc("Gives upper or lower triangular half of a square matrix.");

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/user_ops/triangle.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class Triangle : public OpKernel {
	string Case;
    public:
	    explicit Triangle(OpKernelConstruction* context) : OpKernel(context) {
		    OP_REQUIRES_OK(context, context->GetAttr("Case", &Case));
	    }

	    void Compute(OpKernelContext* context) override {
	    
		    const Tensor & input = context->input(0); 

		    //Check that input represents a matrix.
		    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input.shape()),
		                errors::InvalidArgument("In[0] is not a matrix"));

		    //Check that input matrix is square. 
		    OP_REQUIRES(context, input.dim_size(0) == input.dim_size(1), errors::InvalidArgument("Input matrix must be square."));
		                
		    Tensor* output = NULL;
		    
		    //Allocate space for output
		    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(),
		                                                     &output));
		    
		    if (output->NumElements() == 0) {
		      // the output shape is a 0-element matrix, so there is nothing to do.
		      return;
		    }
		    if (Case=="upper")
		    {
		        // out_mat = in_mat.template triangularView<Eigen::Upper>();    
		        functors::upper_tri<Device, T> triu;
		        triu(context->eigen_device<Device>(), input.flat<T>().data(), output->flat<T>().data(), input.dim_size(0));
		    }
		    else
		    {
		        // out_mat = in_mat.template triangularView<Eigen::Lower>();
		        functors::lower_tri<Device, T> tril;
		        tril(context->eigen_device<Device>(), input.flat<T>().data(), output->flat<T>().data(), input.dim_size(0));           
		    }
	        
	    }
};



REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    Triangle<CPUDevice, float>);
    
REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    Triangle<CPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    Triangle<GPUDevice, float>);
    
REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    Triangle<GPUDevice, double>);
}