import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
import torch.nn.functional as F
from torch import from_numpy
import pickle as pkl
from train_util import *
from .latticesuperpixelcuda.latticesuperpixel import LatticeSuperpixelFunction
from .superpixelcolorcuda.superpixelcolor import SuperpixelColorFunction
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pylab
# define the function includes in import *
__all__ = [
    'SpixelTransNet1l','SpixelTransNet1l_bn'
]

unloader = transforms.ToPILImage()
class SpixelTransNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelTransNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.nobatchNorm = False
        self.conv1 = conv(self.batchNorm, 5, 64, kernel_size=3)
        self.conv2 = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.pool = maxpool(kernel_size = 3, stride = 2, padding = 1)

        self.conv3 = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv4 = conv(self.batchNorm, 64, 64, kernel_size=3)
        #self.pool2 = maxpool(kernel_size = 3, stride = 2, pad = 1)

        self.conv5 = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv6 = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv7 = conv(self.nobatchNorm, 197, 15, kernel_size=3)
        #self.upconv = upsample(self.size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x,K):

        out1 = self.conv2(self.conv1(x)) #11*11
        out2 = self.pool(out1) #23*23
        out3 = self.conv4(self.conv3(out2)) #95*95
        out4 = self.pool(out3)
        out5 = self.conv6(self.conv5(out4))
        out_up6 = F.upsample(out5,[x.size()[-2], x.size()[-1]],mode='bilinear')
        out_up4 = F.upsample(out3,[x.size()[-2], x.size()[-1]],mode='bilinear')
        concat1 = torch.cat((x,out1, out_up4,out_up6), 1)
        out7 = self.conv7(concat1)
        concatall = torch.cat((x,out7), 1)
        prob0 = LatticeSuperpixelFunction.apply(concatall, 4, 4, 1)
            #print(prob0.shape)
        #prob01 = prob0.detach().cpu().numpy()
        #prob01 = prob01.squeeze()
        #print(prob01.shape)
        #np.savetxt('prob01.txt', prob01, fmt='%d')
        #img1 = SuperpixelColorFunction.apply(img, prob0, 2, 2, 2)
        #imgL2 = img1.cpu()
        #print(imgL2.shape)
        #imgL2 = imgL2[0, :, :]
        #print(imgL2.shape)
        #img1 = img1.detach().numpy()
        #np.savetxt('img1.txt', img1[0, :, :], fmt='%.3f')
        #imgL2 = unloader(imgL2)

        # img1 = img1.transpose(0, 1)
        # img1 = img1.transpose(1, 2)
        #plt.imshow(imgL2)
        #pylab.show()
        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def load_weights_from_pkl(self, weights_pkl):
        with open(weights_pkl, 'rb') as wp:
            try:
                # for python3
                name_weights = pkl.load(wp, encoding='latin1')
            except TypeError as e:
                # for python2
                name_weights = pkl.load(wp)
            state_dict = {}

            def _set(layer, key):
                state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])
                state_dict[layer + '.bias'] = from_numpy(name_weights[key]['bias'])

            def _set_bn(layer, key):
                state_dict[layer + '.running_var'] = from_numpy(name_weights[key]['running_var'])
                state_dict[layer + '.running_mean'] = from_numpy(name_weights[key]['running_mean'])
                state_dict[layer + '.weight'] = torch.ones_like(state_dict[layer + '.running_var'])
                state_dict[layer + '.bias'] = torch.zeros_like(state_dict[layer + '.running_var'])

            _set('conv1.0', 'Convolution1')
            _set_bn('conv1.1', 'BatchNorm1')
            _set('conv2.0', 'Convolution2')
            _set_bn('conv2.1', 'BatchNorm2')
            _set('conv3.0', 'Convolution3')
            _set_bn('conv3.1', 'BatchNorm3')
            _set('conv4.0', 'Convolution4')
            _set_bn('conv4.1', 'BatchNorm4')
            _set('conv5.0', 'Convolution5')
            _set_bn('conv5.1', 'BatchNorm5')
            _set('conv6.0', 'Convolution6')
            _set_bn('conv6.1', 'BatchNorm6')
            _set('conv7.0', 'Convolution7')
            self.load_state_dict(state_dict)

def SpixelTransNet1l( data=None):
    # Model without  batch normalization
    model = SpixelTransNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelTransNet1l_bn(data=None):
    # model with batch normalization
    model = SpixelTransNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
#
