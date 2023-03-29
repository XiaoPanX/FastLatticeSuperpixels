#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
  __global__ void PixelFeatureXYRGBForwardGPU(const int nthreads,
    const scalar_t* bottom_data,
    const int height, const int width, const int in_dim, const scalar_t pos_scale,
    const scalar_t color_scale, const scalar_t offset_h, const scalar_t offset_w,
    scalar_t* top_data) {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     if(index < nthreads) {
        const int spatial_dim = height * width;
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;

        const int y = s / width;
        const int x = s % width;

        int out_dim = 2 + in_dim;
        int top_offset_1 = ((n * out_dim) * spatial_dim + s);
        int top_offset_2 = ((n * out_dim + 1) * spatial_dim + s);
        top_data[top_offset_1] = pos_scale * y + offset_h;
        top_data[top_offset_2] = pos_scale * x + offset_w;

        for (unsigned int c = 0; c < in_dim; ++c) {
          int bottom_offset = ((n * in_dim + c) * spatial_dim + s);
          int top_offset = ((n * out_dim + c + 2) * spatial_dim + s);
          top_data[top_offset] = color_scale * bottom_data[bottom_offset];
        }
      }
  }
}
/*template <typename scalar_t>
__global__ void latticesuperpixel_cuda_backward_kernel(
      const int n,
      const int nbatch,
      const int channels,
      const int oheight,
      const int owidth,
      const scalar_t* __restrict__ odata,
      scalar_t* __restrict__ idata) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int spatial_dim = oheight * owidth;
    if (index < n){
          int n1= index/spatial_dim;
          int n2= index%spatial_dim;
          for (int c = 0; c < channels; ++c) {
   		        int64_t pindex = (n1*channels+c)*spatial_dim+n2;
                idata[pindex] = odata[pindex];
          }
     }
  }
}*/
at::Tensor pixelfeature_cuda_forward(
    const at::Tensor input,
    const float pos_scale,
    const float color_scale) {
  auto output = at::empty_like(input);
  const int height = input.size(2);
  const int width = input.size(3);
  const int num_ = input.size(0);
  const int channels = input.size(1);
  output.resize_({num_, 2+channels, height, width});
  const float offset_h = 0.0;
  const float offset_w = 0.0;
  output.contiguous();
    const int totalthreads = num_ * height * width;
    int num_threads = 512;
    int num_kernels = totalthreads/num_threads+1;
    // NOLINT_NEXT_LINE(whitespace/operators)
   AT_DISPATCH_FLOATING_TYPES(input.type(), "pixelfeature_forward_cuda", ([&] {
    PixelFeatureXYRGBForwardGPU<scalar_t><<<num_kernels, num_threads>>>(
      totalthreads,
      input.data<scalar_t>(),
      height,
      width,
      channels,
      pos_scale,
      color_scale,
      offset_h,
      offset_w,
      output.data<scalar_t>());
   }));
  return output;
}

at::Tensor pixelfeature_cuda_backward(const at::Tensor& grad_output){

  return grad_output;
}
