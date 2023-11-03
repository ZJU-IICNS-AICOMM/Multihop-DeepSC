
import math
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

def cal_nmse(sparse_gt, sparse_pred):
    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    return nmse

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

def F_norm(H):
    H = H[:,:,0]+1j*H[:,:,1]
    return np.expand_dims(np.sqrt((np.mean(abs(H**2),axis=(1,2),keepdims=True))),3)

def pwr_normalize(sig):
    _, num_ele = sig.shape[0], torch.numel(sig[0])
    pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
    sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))
    return sig

def power_norm(signal, power=1):
    num_elements = len(signal.flatten())
    num_complex = num_elements//2//2
    temp = signal.view(num_complex, 2, 1, 2)
    signal_power = torch.mean(temp[:,0]**2+temp[:,1]**2)
    signal = signal * math.sqrt(power) / torch.sqrt(signal_power)
    return signal

def power_norm_batchwise(signal, power=1):
    batchsize , num_elements = signal.shape[0], len(signal[0].flatten())
    num_complex = num_elements//2 
    signal_shape = signal.shape
    signal = signal.view(batchsize, num_complex, 2)
    
    # print(torch.mean((signal[:,:,0]**2 + signal[:,:,1]**2)))

    signal_power = torch.sum((signal[:,:,0]**2 + signal[:,:,1]**2), dim=-1)/num_complex

    signal = signal * math.sqrt(power) / torch.sqrt(signal_power.unsqueeze(-1).unsqueeze(-1))
    
    signal = signal.view(signal_shape)
    # print(torch.mean((signal[:,:,0]**2 + signal[:,:,1]**2)))
    return signal

def noise_gen(is_train):
    min_snr, max_snr = -6, 18
    diff_snr = max_snr - min_snr
    
    min_var, max_var = 10**(-min_snr/20), 10**(-max_snr/20)
    diff_var = max_var - min_var
    if is_train:
        channel_snr = torch.FloatTensor([18])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    else:
        channel_snr = torch.FloatTensor([18])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    return channel_snr, noise_var 


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    


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


class Generator(nn.Module):
    """Generator for denoising"""
    def __init__(self, middle_kernel=64):
        super(Generator, self).__init__()
        # encoding blocks
        self.conv1 = Conv(3, middle_kernel, 4)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv3 = Conv(middle_kernel * 2, middle_kernel * 4, 4)

        # residual blocks
        self.conv4 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)
        self.conv5 = Conv(middle_kernel * 4, middle_kernel * 4, 3, 1, 1)

        # decoding blocks
        self.Tconv1 = TConv(middle_kernel * 4, middle_kernel * 2, 4)
        self.Tconv2 = TConv(middle_kernel * 2, middle_kernel, 4)
        self.Tconv3 = TConv(middle_kernel, 3, 4, batchnorm=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  
        out = F.leaky_relu(self.conv2(out), 0.05)  
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)  
        out = F.leaky_relu(self.conv5(out), 0.05) 

        out = F.leaky_relu(self.Tconv1(out), 0.05)
        out = F.leaky_relu(self.Tconv2(out), 0.05)  
        out = torch.tanh(self.Tconv3(out))  
        return out

class Discriminator(nn.Module):
    """Discriminator for raw data."""
    def __init__(self, middle_kernel=64):
        super(Discriminator, self).__init__()
        self.conv1 = Conv(3, middle_kernel, 4, batchnorm=False)
        self.conv2 = Conv(middle_kernel, middle_kernel * 2, 4)
        self.conv3 = Conv(middle_kernel * 2, middle_kernel * 4, 4)
        self.fc = Conv(middle_kernel*4, 1, 4, 1, 0, False)
   
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  
        out = F.leaky_relu(self.conv2(out), 0.05) 
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = self.fc(out).squeeze()
        return out

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

