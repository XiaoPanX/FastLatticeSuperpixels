#include <torch/torch.h>
#include <vector>

at::Tensor latticesuperpixel_cuda_forward(
    at::Tensor input,
    int seed_h,
    int seed_w,
    int seed_level);

at::Tensor latticesuperpixel_cuda_backward(
   const at::Tensor& grad_outpu);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor latticesuperpixel_forward(
    at::Tensor input,
    int seed_h,
    int seed_w,
    int seed_level) {
  //CHECK_INPUT(input1);
  return latticesuperpixel_cuda_forward(input, seed_h,seed_w, seed_level);
}

at::Tensor latticesuperpixel_backward(
const at::Tensor& grad_output){
    return latticesuperpixel_cuda_backward(grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &latticesuperpixel_forward, "latticesuperpixel forward (CUDA)");
  m.def("backward", &latticesuperpixel_backward, "latticesuperpixel backward (CUDA)");
}
