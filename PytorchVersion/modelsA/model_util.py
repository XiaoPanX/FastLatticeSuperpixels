import torch
import torch.nn as nn

## *************************** my functions ****************************

def predict_param(in_planes, channel=3):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, channel=9):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_feat(in_planes, channel=20, stride=1):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)

def predict_prob(in_planes, channel=9):
    return  nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )
#***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )
def maxpool(kernel_size, stride=None, padding=0):
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


def upsample(size, mode='bilinear',align_corners=False):
    return nn.Upsample(size=size, mode=mode,align_corners=align_corners)

def crop(scale_factor, mode='bilinear',align_corners=False):
    return nn.Upsample(scale_factor=scale_factor, mode=mode,align_corners=align_corners)



