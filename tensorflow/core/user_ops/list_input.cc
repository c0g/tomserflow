#include "tensorflow/core/framework/op.h"

REGISTER_OP("KvsDotVec")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type");

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"


namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class KvsDotVec : public OpKernel {
    public:
	    explicit KvsDotVec(OpKernelConstruction* context) : OpKernel(context) {};

	    void Compute(OpKernelContext* ctx) override {
	    
	        OpInputList values;
    		OP_REQUIRES_OK(ctx, ctx->input_list("values", &values));
    		 const int N = values.size();
    		 std::cout << N << std::endl;
	    }
};

REGISTER_KERNEL_BUILDER(
    Name("KvsDotVec")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
    KvsDotVec<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("KvsDotVec")
        .Device(DEVICE_CPU)
        .TypeConstraint<double>("T"),
    KvsDotVec<CPUDevice, double>);

}