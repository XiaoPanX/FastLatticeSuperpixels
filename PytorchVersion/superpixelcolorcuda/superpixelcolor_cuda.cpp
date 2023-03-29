#include <torch/torch.h>
#include <vector>

at::Tensor superpixelcolor_cuda_forward(
    at::Tensor input1,
    at::Tensor input2,
     int seed_h,
    int seed_w,
    int seed_level);

at::Tensor superpixelcolor_cuda_backward(
   const at::Tensor& grad_output);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor superpixelcolor_forward(
    at::Tensor input1,
    at::Tensor input2,
    int seed_h,
    int seed_w,
    int seed_level) {
  //CHECK_INPUT(input1);
  return superpixelcolor_cuda_forward(input1,input2, seed_h,seed_w,seed_level);
}

at::Tensor superpixelcolor_backward(
const at::Tensor& grad_output){
    return superpixelcolor_cuda_backward(grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &superpixelcolor_forward, "superpixelcolor forward (CUDA)");
  m.def("backward", &superpixelcolor_backward, "superpixelcolor backward (CUDA)");
}
