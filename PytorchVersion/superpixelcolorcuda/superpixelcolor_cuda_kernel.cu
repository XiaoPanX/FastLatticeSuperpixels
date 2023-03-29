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

template <typename scalar_t>
__global__ void get_Total_Color(const int nthreads, const scalar_t* image_data,  const scalar_t* labels, scalar_t *Av_color, scalar_t *supersize,
int spatial_dim,int label_dim, int channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads){
        int n = index/spatial_dim;
        int s = index%(spatial_dim);
        int top_level_index=n*spatial_dim;
        int pixel_label=labels[index];
        for(int k=0;k<channels;k++)
        {
         //  int k1 = abs(k-2);
            int out_index=(n*channels+k)*label_dim+pixel_label;
            int image_index = (n*channels+k)*spatial_dim+s;
          //  int image_index1 = (n*3+k1)*spatial_dim+s;
          //  Av_color[out_index]+=image_data[image_index];
            atomicAdd(Av_color+out_index, image_data[image_index]);

        }
        int numindex = n*label_dim+pixel_label;
        atomicAdd(supersize+numindex, 1.0);
    }
  }
 template <typename scalar_t>
__global__ void get_AV_Color(const int nthreads, const scalar_t* T,  scalar_t *Av_color,const int nr_seeds_h1, const int nr_seeds_w1,
int label_dim, int channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
        int n = index/label_dim;
        int s = index%(label_dim);
        int T_index=index;
        for(int k=0;k<channels;k++){
            int out_index=(n*channels+k)*label_dim+s;
            Av_color[out_index]=Av_color[out_index]/(float)(T[T_index]);
        }
    }
}
}
/*template <typename scalar_t>
__global__ void superpixelcolor_cuda_backward_kernel(
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
at::Tensor superpixelcolor_cuda_forward(
    const at::Tensor input1,
    const at::Tensor input2,
    const int seed_h,
    const int seed_w,
    const int seed_level) {

  const int height = input1.size(2);
  const int width = input1.size(3);
  const int num_ = input1.size(0);
  const int channels = input1.size(1);
  int seeds_h = seed_h;
  int seeds_w = seed_w;
  if(height>width)
  {
   int temp = seeds_h;
   seeds_h = seeds_w;
   seeds_w = temp;
  }
  int nr_seeds_w = floor(float(float(width)/float(seeds_w))+0.5);
  int nr_seeds_h = floor(float(float(height)/float(seeds_h))+0.5);
  for( int level=1;level<seed_level;level++){
   nr_seeds_w = floor(float(nr_seeds_w/2.0));
   nr_seeds_h = floor(float(nr_seeds_h/2.0));
   }
  auto output = at::empty_like(input1);
  output.resize_({num_, channels, nr_seeds_h, nr_seeds_w});
  auto supsize = at::empty_like(input1);
  supsize.resize_({num_, 1, nr_seeds_h, nr_seeds_w});
  output.contiguous();
  output.zero_();
  supsize.contiguous();
  supsize.zero_();
    const int totalthreads = num_ * height * width;
    int num_threads = 512;
    int num_kernels = totalthreads/num_threads+1;
    int spatial_dim = height*width;
    int label_dim = nr_seeds_h * nr_seeds_w;
    // NOLINT_NEXT_LINE(whitespace/operators)
   AT_DISPATCH_FLOATING_TYPES(input1.type(), "superpixelcolor_forward_cuda", ([&] {
    get_Total_Color<scalar_t><<<num_kernels, num_threads>>>(
      totalthreads,
      input1.data<scalar_t>(),
      input2.data<scalar_t>(),
      output.data<scalar_t>(),
      supsize.data<scalar_t>(),
      spatial_dim,
      label_dim,
      channels);
   }));
   const int totalthreads1 = num_ * nr_seeds_w * nr_seeds_h;
    int num_kernels1 = totalthreads1/num_threads+1;
   AT_DISPATCH_FLOATING_TYPES(input1.type(), "superpixelcolor_forward_cuda", ([&] {
    get_AV_Color<scalar_t><<<num_kernels1, num_threads>>>(
      totalthreads1,
      supsize.data<scalar_t>(),
      output.data<scalar_t>(),
      nr_seeds_h, nr_seeds_w,
      label_dim,
      channels);
   }));
  return output;
}

at::Tensor superpixelcolor_cuda_backward(const at::Tensor& grad_output){

  /*const int height = grad_output.size(2);
  const int width = grad_output.size(3);
  const int num_ = grad_output.size(0);
  const int channels = grad_output.size(1);
  auto newgrad = at::empty_like(input1);
  newgrad.resize_({num_, channels, height, width});
  newgrad.zero_();*/
  return grad_output;
}
