import torch
from torch.nn import Module, Parameter
from torch.autograd import Function
import numpy as np
import torch

import pixelfeature_cuda


class PixelFeatureFunction(Function):
    
    @staticmethod
    def forward(ctx, input, pos_scale, color_scale):
        output = pixelfeature_cuda.forward(input, pos_scale, color_scale)
        #grad_input = upsuperpixel_cuda.backward(superlabel, input, superlabel)
        #height = input.size()
        #input_size1 = input.size()
        #arr = np.array([input_size1[2], input_size1[3]])
        #input_size=torch.from_numpy(arr)
        #print(input_size.shape)
        #ctx.save_for_backward(input, seed_h, seed_w,seed_level)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        #print (grad_output.shape)
        grad_iutput = pixelfeature_cuda.backward(grad_output)
        return grad_iutput
        
class PixelSuperpixel(Module):

    def __init__(self, pos_scale, color_scale):
        super(PixelSuperpixel, self).__init__()
       # self.input_size = input_size
        self.pos_scale = pos_scale
        self.color_scale = color_scale


    def forward(self, input):
        return PixelFeatureFunction.apply(input, self.pos_scale, self.color_scale)
