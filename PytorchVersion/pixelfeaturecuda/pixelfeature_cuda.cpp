#include <torch/torch.h>
#include <vector>

at::Tensor pixelfeature_cuda_forward(
    at::Tensor input,
    float pos_scale,
    float color_scale);

at::Tensor pixelfeature_cuda_backward(
   const at::Tensor& grad_outpu);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor pixelfeature_forward(
    at::Tensor input,
    float pos_scale,
    float color_scale) {
  //CHECK_INPUT(input1);
  return pixelfeature_cuda_forward(input, pos_scale,color_scale);
}

at::Tensor pixelfeature_backward(
const at::Tensor& grad_output){
    return pixelfeature_cuda_backward(grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pixelfeature_forward, "pixelfeature forward (CUDA)");
  m.def("backward", &pixelfeature_backward, "pixelfeature backward (CUDA)");
}
