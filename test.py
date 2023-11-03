import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from collections import OrderedDict


def TConv(in_planes, out_planes, kernel_size, stride=2, padding=1, batchnorm=True): 
    """Tconvolutional layer"""
    layers = [nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_planes))
    return nn.Sequential(*layers)


def Conv(in_planes, out_planes, kernel_size, stride=2, padding=1, batchnorm=True): 
    """Convolutional layer"""
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_planes))
    return nn.Sequential(*layers)


class GA2B(nn.Module):
    """Generator for denoising"""
    def __init__(self, middle_kernel=64):
        super(GA2B, self).__init__()
        # encoding blocks
        self.conv1 = Conv(3, middle_kernel, 4)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv2_2 = Conv(middle_kernel * 2, middle_kernel * 4, 4)

        # residual blocks
        self.conv3 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)
        self.conv4 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)

        # decoding blocks
        self.deconv1_1 = TConv(middle_kernel * 4, middle_kernel * 2, 4)
        self.deconv1 = TConv(middle_kernel * 2, middle_kernel, 4)
        self.deconv2 = TConv(middle_kernel, 3, 4, batchnorm=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  
        out = F.leaky_relu(self.conv2(out), 0.05)  
        out = F.leaky_relu(self.conv2_2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)  
        out = F.leaky_relu(self.conv4(out), 0.05) 

        out = F.leaky_relu(self.deconv1_1(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  
        out = torch.tanh(self.deconv2(out))  
        return out


class GB2A(nn.Module):
    """Generator for recovering"""

    def __init__(self, middle_kernel=64):
        super(GB2A, self).__init__()
        # encoding blocks
        self.conv1 = Conv(3, middle_kernel, 4)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv2_2 = Conv(middle_kernel * 2, middle_kernel * 4, 4)

        self.conv3 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)
        self.conv4 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)

        # decoding blocks
        self.deconv1_1 = TConv(middle_kernel * 4, middle_kernel * 2, 4)
        self.deconv1 = TConv(middle_kernel * 2, middle_kernel, 4)
        self.deconv2 = TConv(middle_kernel, 3, 4, batchnorm=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  
        out = F.leaky_relu(self.conv2(out), 0.05)  
        out = F.leaky_relu(self.conv2_2(out), 0.05)

        out = F.leaky_relu(self.conv3(out), 0.05)  
        out = F.leaky_relu(self.conv4(out), 0.05)  

        out = F.leaky_relu(self.deconv1_1(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05) 
        out = torch.tanh(self.deconv2(out))  
        return out
    
     
class DA(nn.Module):
    """Discriminator for raw data."""

    def __init__(self, middle_kernel=64, use_labels=False):
        super(DA, self).__init__()
        self.conv1 = Conv(3, middle_kernel, 4, batchnorm=False)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv3 = Conv(middle_kernel * 2, middle_kernel * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = Conv(middle_kernel*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  
        out = F.leaky_relu(self.conv2(out), 0.05) 
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = self.fc(out).squeeze()
        return out


class DB(nn.Module):
    """Discriminator for noisy data."""

    def __init__(self, middle_kernel=64, use_labels=True):
        super(DB, self).__init__()
  
        self.conv1 = Conv(3, middle_kernel, 4, batchnorm=False)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv3 = Conv(middle_kernel * 2, middle_kernel * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = Conv(middle_kernel*4, n_out, 4, 1, 0, False)
    
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) 
        out = F.leaky_relu(self.conv2(out), 0.05) 
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = self.fc(out).squeeze()
        return out



model =  DB(middle_kernel=64)
# print(model)
x = torch.ones([2,3,32,32]) + 0.

ouputs = model(x)
print(ouputs.shape)