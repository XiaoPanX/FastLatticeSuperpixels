#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""
import numpy as np
from init_caffe import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile
from loss_functions import *

trans_dim = 15

def normalize(bottom, dim):

    bottom_relu = L.ReLU(bottom)
    sum = L.Convolution(bottom_relu,
                        convolution_param = dict(num_output = 1, kernel_size = 1, stride = 1,
                                                 weight_filler = dict(type = 'constant', value = 1),
                                                 bias_filler = dict(type = 'constant', value = 0)),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])

    denom = L.Power(sum, power=(-1.0), shift=1e-12)
    denom = L.Tile(denom, axis=1, tiles=dim)

    return L.Eltwise(bottom_relu, denom, operation=P.Eltwise.PROD)

def conv_bn_relu_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param = dict(num_output = num_out, kernel_size = 3, stride = 1, pad = 1,
                                                   weight_filler = dict(type = 'gaussian', std = 0.001),
                                                   bias_filler = dict(type = 'constant', value = 0)),
                                                   # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    bn1 = L.BatchNorm(conv1)
    bn1 = L.ReLU(bn1, in_place = True)

    return bn1

def conv_relu_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param = dict(num_output = num_out, kernel_size = 3, stride = 1, pad = 1,
                                                   weight_filler = dict(type = 'gaussian', std = 0.001),
                                                   bias_filler = dict(type = 'constant', value = 0)),
                                                   # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    conv1 = L.ReLU(conv1, in_place = True)

    return conv1

def cnn_module(bottom, num_out):

    conv1 = conv_bn_relu_layer(bottom, 64)
    conv2 = conv_bn_relu_layer(conv1, 64)
    pool1 = L.Pooling(conv2, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    conv3 = conv_bn_relu_layer(pool1, 64)
    conv4 = conv_bn_relu_layer(conv3, 64)
    pool2 = L.Pooling(conv4, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    conv5 = conv_bn_relu_layer(pool2, 64)
    conv6 = conv_bn_relu_layer(conv5, 64)

   # pool3 = L.Pooling(conv6, pooling_param=dict(kernel_size=3, stride=2, pad=1, pool=P.Pooling.MAX))
   # conv7 = conv_bn_relu_layer(pool3, 64)
   # conv8 = conv_bn_relu_layer(conv7, 64)

   # conv8_upsample=L.Interp(conv8, interp_param = dict(zoom_factor = 8))
   # conv8_upsample_crop = L.Crop(conv8_upsample, conv2)

    conv6_upsample = L.Interp(conv6, interp_param = dict(zoom_factor = 4))
    conv6_upsample_crop = L.Crop(conv6_upsample, conv2)

    conv4_upsample = L.Interp(conv4, interp_param = dict(zoom_factor = 2))
    conv4_upsample_crop = L.Crop(conv4_upsample, conv2)

    conv_concat = L.Concat(bottom, conv2, conv4_upsample_crop, conv6_upsample_crop)
  #  conv_concat2 = L.Concat(conv4_upsample_crop, conv6_upsample_crop)
    conv7 = conv_relu_layer(conv_concat, num_out)
 #   HighFeature = conv_relu_layer(conv_concat2, num_out)
    conv_comb = L.Concat(bottom, conv7)

    return conv_comb


def compute_assignments(spixel_feat, pixel_features,
                        spixel_init, num_spixels_h,
                        num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)

    pixel_spixel_neg_dist = L.Passoc(pixel_features, spixel_feat, spixel_init,
                                     spixel_feature2_param =\
          dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w, scale_value = -1.0))

    # Softmax to get pixel-superpixel relative soft-associations
    pixel_spixel_assoc = L.Softmax(pixel_spixel_neg_dist)

    return pixel_spixel_assoc


def compute_final_spixel_labels(pixel_spixel_assoc,
                                spixel_init,
                                num_spixels_h, num_spixels_w):

    # Compute new spixel indices
    rel_label = L.ArgMax(pixel_spixel_assoc, argmax_param = dict(axis = 1),
                         propagate_down = False)
    new_spix_indices = L.RelToAbsIndex(rel_label, spixel_init,
                                       rel_to_abs_index_param = dict(num_spixels_h = int(num_spixels_h),
                                                                     num_spixels_w = int(num_spixels_w)),
                                                                     propagate_down = [False, False])

    return new_spix_indices


def decode_features(pixel_spixel_assoc, spixel_feat, spixel_init,
                    num_spixels_h, num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)

    # Reshape superpixel features to k_h x k_w
    spixel_feat_reshaped = L.Reshape(spixel_feat,
                                      reshape_param = dict(shape = {'dim':[0,0,num_spixels_h,num_spixels_w]}))

    # Concatenate neighboring superixel features
    concat_spixel_feat = L.Convolution(spixel_feat_reshaped,
                                        name = 'concat_spixel_feat_' + str(num_channels),
                                        convolution_param = dict(num_output = num_channels * 9,
                                                                 kernel_size = 3,
                                                                 stride = 1,
                                                                 pad = 1,
                                                                 group = num_channels,
                                                                 bias_term = False),
                                                                 param=[{'name': 'concat_spixel_feat_' + str(num_channels),
                                                                        'lr_mult':0, 'decay_mult':0}])

    # Spread features to pixels
    flat_concat_label = L.Reshape(concat_spixel_feat,
                                  reshape_param = dict(shape = {'dim':[0, 0, 1, num_spixels]}))
    img_concat_spixel_feat = L.Smear(flat_concat_label, spixel_init)

    tiled_assoc = L.Tile(pixel_spixel_assoc,
                         tile_param = dict(tiles = num_channels))

    weighted_spixel_feat = L.Eltwise(img_concat_spixel_feat, tiled_assoc,
                                     eltwise_param = dict(operation = P.Eltwise.PROD))
    recon_feat = L.Convolution(weighted_spixel_feat,
                               name = 'recon_feat_' + str(num_channels),
                               convolution_param = dict(num_output = num_channels,
                                                        kernel_size = 1,
                                                        stride = 1,
                                                        pad = 0,
                                                        group = num_channels,
                                                        bias_term = False,
                                                        weight_filler = dict(type = 'constant', value = 1.0)),
                                                        param=[{'name': 'recon_feat_' + str(num_channels),
                                                               'lr_mult':0, 'decay_mult':0}])

    return recon_feat


def exec_iter(spixel_feat, trans_features, spixel_init,
              num_spixels_h, num_spixels_w, num_spixels,
              trans_dim):

    # Compute pixel-superpixel assignments
    pixel_assoc = \
        compute_assignments(spixel_feat, trans_features,
                            spixel_init, num_spixels_h,
                            num_spixels_w, num_spixels, trans_dim)
    # Compute superpixel features from pixel assignments
    spixel_feat1 = L.SpixelFeature2(trans_features,
                                    pixel_assoc,
                                    spixel_init,
                                    spixel_feature2_param =\
        dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))

    return spixel_feat1


def create_ssn_net(img_height, img_width,
                    pos_scale, color_scale,
                   s_h, s_w, s_l,
                   phase = None):

    n = caffe.NetSpec()

    n.img = L.Input(shape=[dict(dim=[1, 3, img_height, img_width])])


    n.pixel_features = L.PixelFeature(n.img,
                                      pixel_feature_param = dict(type = P.PixelFeature.POSITION_AND_RGB,
                                                                 pos_scale = float(pos_scale),
                                                                 color_scale = float(color_scale)))

    ### Transform Pixel features
    n.trans_features = cnn_module(n.pixel_features, trans_dim)
    n.new_spix_indices, n.avcolor = L.LatticeSpixel(n.trans_features, n.img, lattice_spixel_param= \
        dict(seeds_h=int(s_h), seeds_w=int(s_w), nr_levels=int(s_l)), ntop=2)


    return n.to_proto()


def load_ssn_net(img_height, img_width,
                  pos_scale, color_scale,
                  s_h, s_w, s_l):

    net_proto = create_ssn_net(img_height, img_width,
                               pos_scale, color_scale,
                               s_h, s_w, s_l)
   # f1 = open("./lib/caffe/models/test.prototxt", "w+")
 #   f1.write(str(net_proto))
 #   f1.close()
    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    print('*******')
    print(f.name)
    print('*******')
    return caffe.Net(f.name, caffe.TEST)


def get_ssn_net(img_height, img_width,
                num_spixels, pos_scale, color_scale,
                num_spixels_h, num_spixels_w, num_steps,
                phase):

    # Create the prototxt
    net_proto = create_ssn_net(img_height, img_width,
                               num_spixels, pos_scale, color_scale,
                               num_spixels_h, num_spixels_w, int(num_steps), phase)
 #   f1 = open("./lib/caffe/models/train.prototxt", "w+")
#    f1.write(str(net_proto))
  #  f1.close()
    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()

    return f.name
