#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
import scipy.io as sio
import os
import scipy
from scipy.misc import fromimage
from scipy.misc import imsave
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import pylab
import linecache
from init_caffe import *
from config import *
from utils import *
from fetch_and_transform_data import fetch_and_transform_data, transform_and_get_spixel_init
from creat_net_test import load_ssn_net
from skimage.color import lab2rgb
import sys
import time
sys.path.append('../lib/cython')
from connectivity import enforce_connectivity

def compute_spixels(data_type, n_spixels,
                    caffe_model, out_folder):
    image_list = IMG_LIST[data_type]
    label_folder = out_folder + '/label/'
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    image_folder = out_folder + '/imgBdy/'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    #p_scale = 0.40
    p_scale = 0.4
    color_scale = 0.26
    meantime = 0.0;
    numimg = 0;
    s_h = 2; ##
    s_w = 2;
    s_l = 4;
    with open(image_list) as list_f:
        name_list = list_f.readlines()
        for imgname in name_list:
            print(imgname)
            imgname = imgname[:-1]
            [inputs, height, width] = \
                fetch_and_transform_data(imgname, data_type,
                                         ['img', 'label', 'problabel'],
                                         int(n_spixels))

            height = inputs['img'].shape[2]
            width = inputs['img'].shape[3]
            [spixel_initmap, feat_spixel_initmap, num_spixels_h, num_spixels_w] =\
                	transform_and_get_spixel_init(int(n_spixels), [height, width])

            dinputs = {}
            dinputs['img'] = inputs['img']
            #dinputs['spixel_init'] = spixel_initmap
            #dinputs['feat_spixel_init'] = feat_spixel_initmap

            pos_scale_w = (1.0 * num_spixels_w) / (float(p_scale) * width)
            pos_scale_h = (1.0 * num_spixels_h) / (float(p_scale) * height)
            pos_scale = np.max([pos_scale_h, pos_scale_w])
            net = load_ssn_net(height, width,
                               pos_scale, color_scale,
                               s_h, s_w, s_l)

            if caffe_model is not None:
                	net.copy_from(caffe_model)
            else:
                net = initialize_net_weight(net)
                print('error!!')

            num_spixels = int(num_spixels_w * num_spixels_h)
            result = net.forward_all(**dinputs)

            given_img = fromimage(Image.open(IMG_FOLDER[data_type] + imgname + '.jpg'))
            spix_index = np.squeeze(net.blobs['new_spix_indices'].data).astype(int)
            spixel_image = get_spixel_image(given_img, spix_index)
            out_img_file = image_folder + imgname + '_bdry.jpg'
            imsave(out_img_file, spixel_image)
            out_file = label_folder + imgname + '.txt'
            np.savetxt(out_file, spix_index, fmt='%d')

    return



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', default='TEST', type=str)
    parser.add_argument('--n_spixels', default=600, type=int)
    parser.add_argument('--caffemodel', type=str, default='./models/bsd500.caffemodel')
    parser.add_argument('--result_dir', type=str, default='./result/600/')

    var_args = parser.parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(1)
    compute_spixels(var_args.datatype,
                    var_args.n_spixels,
                    var_args.caffemodel,
                    var_args.result_dir)

if __name__ == '__main__':
    main()
