#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/layers/lattice_spixel_layer.hpp"

namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void LatticeSpixelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  in_channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  LatticeSpixelParameter lattice_spixel_param = this->layer_param_.lattice_spixel_param();

  seeds_h_ = lattice_spixel_param.seeds_h();
  seeds_w_ = lattice_spixel_param.seeds_w();
  nr_levels_= lattice_spixel_param.nr_levels();
  top[0]->Reshape(num_, 1, height_, width_);
  if(height_>width_)
  {
   int temp = seeds_h_;
   seeds_h_ = seeds_w_;
   seeds_w_ = temp;
  }
  int nr_seeds_w = floor(float(float(width_)/float(seeds_w_))+0.5);
  int nr_seeds_h = floor(float(float(height_)/float(seeds_h_))+0.5);
  //int nr_seeds_w = floor(float(width_/seeds_w_)+0.5);
  //int nr_seeds_h = floor(float(height_/seeds_h_)+0.5);
  get_avcolor_ = false;
  if (top.size() > 1) {
    get_avcolor_= true;
    int nr_seeds_ww=nr_seeds_w;
    int nr_seeds_hh=nr_seeds_h;
    for( int ll=1;ll<nr_levels_;ll++){
       nr_seeds_ww = floor(float(nr_seeds_ww/2.0));
       nr_seeds_hh = floor(float(nr_seeds_hh/2.0));
        }
    top[1]->Reshape(num_,3,nr_seeds_hh,nr_seeds_ww);
  }
  labels_.Reshape(num_,nr_levels_,height_, width_);
  parents_.Reshape(num_,nr_levels_,nr_seeds_h,nr_seeds_w);
  nr_partitions_.Reshape(num_,nr_levels_,nr_seeds_h,nr_seeds_w);
  //nr_labels_.Reshape(num_,nr_levels_,1,2);
  Block_feature_.Reshape(num_,(nr_levels_)*in_channels_,nr_seeds_h,nr_seeds_w);
  T_.Reshape(num_,nr_levels_,nr_seeds_h,nr_seeds_w);
  pixel_Tag_.Reshape(num_,1,height_,width_);
  block_Tag_.Reshape(num_,1,nr_seeds_h,nr_seeds_w);
}

template <typename Dtype>
void LatticeSpixelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, 1, height_, width_);
   /* if (top.size() > 1) {
      top[1]->Reshape(num_, out_channels_, height_, width_);
    }*/
}

/*
Forward CPU function
*/
template <typename Dtype>
void LatticeSpixelLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

/*
Backward CPU function (NOT_IMPLEMENTED for now)
 */
template <typename Dtype>
void LatticeSpixelLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(LatticeSpixelLayer);
#endif

INSTANTIATE_CLASS(LatticeSpixelLayer);
REGISTER_LAYER_CLASS(LatticeSpixel);

}  // namespace caffe
