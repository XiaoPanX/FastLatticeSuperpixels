import torch
from torch.nn import Module, Parameter
from torch.autograd import Function
import numpy as np
import torch

import latticesuperpixel_cuda


class LatticeSuperpixelFunction(Function):
    
    @staticmethod
    def forward(ctx, input, seed_h, seed_w,seed_level):
        output = latticesuperpixel_cuda.forward(input, seed_h, seed_w,seed_level)
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
        grad_iutput = latticesuperpixel_cuda.backward(grad_output)
        return grad_iutput
        
class LatticeSuperpixel(Module):

    def __init__(self, seed_h, seed_w,seed_level):
        super(LatticeSuperpixel, self).__init__()
       # self.input_size = input_size
        self.seed_h = seed_h
        self.seed_w = seed_w
        self.seed_level = seed_level

    def forward(self, input):
        return LatticeSuperpixelFunction.apply(input, self.seed_h, self.seed_w, self.seed_level)
