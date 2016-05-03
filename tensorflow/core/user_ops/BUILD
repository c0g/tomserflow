load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "get_diag.so",
    srcs = ["get_diag.cc", "get_diag.h"],
    gpu_srcs = ["get_diag.cu.cc", "get_diag.h", "get_diag_gpu.h"]
)

tf_custom_op_library(
    name = "triangle.so",
    srcs = ["triangle.cc", "triangle.h"],
    gpu_srcs = ["triangle.cu.cc", "triangle.h", "triangle_gpu.h"]
)

tf_custom_op_library(
    name = "gpu_cholgrad.so",
    srcs = ["gpu_cholgrad.cc", "gpu_cholgrad.h"],
    gpu_srcs = ["gpu_cholgrad.cu.cc", "gpu_cholgrad.h", "gpu_cholgrad_gpu.h"]
)

tf_custom_op_library(
    name = "gpu_cholesky.so",
    srcs = ["gpu_cholesky.cc", "gpu_cholesky.h", "linalg_ops_common.h"],
    gpu_srcs = ["gpu_cholesky.cu.cc", "gpu_cholesky.h", 
    			"gpu_cholesky_gpu.h", "triangle_gpu.h", "triangle.h"],
)