import torch
from torch.nn import Module, Parameter
from torch.autograd import Function
import numpy as np
import torch

import superpixelcolor_cuda


class SuperpixelColorFunction(Function):
    
    @staticmethod
    def forward(ctx, input,suplabel, seed_h, seed_w,seed_level):
        output = superpixelcolor_cuda.forward(input,suplabel, seed_h, seed_w,seed_level)
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
        grad_iutput = superpixelcolor_cuda.backward(grad_output)
        return grad_iutput
        
class SuperpixelColor(Module):

    def __init__(self, suplabel, seed_h, seed_w,seed_level):
        super(SuperpixelColor, self).__init__()
       # self.input_size = input_size
        self.superlabel = suplabel
        self.seed_h = seed_h
        self.seed_w = seed_w
        self.seed_level = seed_level


    def forward(self, input):
        return SuperpixelColorFunction.apply(input, self.superlabel, self.seed_h, self.seed_w, self.seed_level)
