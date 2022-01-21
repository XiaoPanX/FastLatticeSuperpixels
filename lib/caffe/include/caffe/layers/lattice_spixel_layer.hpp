/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#ifndef Lattice_Spixel_LAYER_HPP_
#define Lattice_Spixel_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  template <typename Dtype>
  class LatticeSpixelLayer : public Layer<Dtype> {
   public:
    explicit LatticeSpixelLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "LatticeSpixel"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_;
    int in_channels_;
    int height_;
    int width_;
    int seeds_w_;
    int seeds_h_;
    int nr_levels_;
    bool get_avcolor_;
   // int num_spixels_w_;

    Blob<int> labels_;
    Blob<int> parents_;
    Blob<int> nr_partitions_;
  //  Blob<int> nr_labels_;
    Blob<int> pixel_Tag_;
    Blob<int> block_Tag_;
    Blob<Dtype> Block_feature_;
    Blob<int> T_;
   // Blob<Dtype> dis_;

  };

} //namespace caffe

#endif  // SPIXEL_FEATURE_LAYER_HPP_
