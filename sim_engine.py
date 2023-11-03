import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import math
import argparse

from utils import *

from model_utils import *
from einops import rearrange

 

# Testing the model
def test_deepsc_tx(net, device, dataloader, criterion, sample_num):
    net.eval()
    tx_time = 30
    snr = 18
    PSNR_Group = np.zeros(tx_time)
    with torch.no_grad():
        psnr_list = []
        print('Test SNR = %d dB' % (snr) )
        test_loss = 0
        MSE_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # if batch_idx==20:
            #     break
            inputs, targets = inputs.to(device), inputs.to(device)
            for tx in range(tx_time):
                outputs = net(inputs, snr=snr)        
                       
                loss = criterion(outputs, targets)                              
                test_loss = loss.item()
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                target = torch.chunk(targets, chunks=inputs.size(0), dim=0)
                
                ######  PSNR/SSIM/etc  ######
                psnr_vals = calc_psnr(predictions, target)
                PSNR = torch.mean(torch.tensor(psnr_vals)).item()

                inputs = outputs[:]
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Mse: %.3f | PSNR: %.2f'
                            % (test_loss, MSE_loss/(batch_idx+1), PSNR))
                PSNR_Group[tx] += PSNR
        PSNR_Group /= (batch_idx+1) 
    return PSNR_Group


##  This function is used to test the scheme in different SNRs
def Experiment_1(net, device, testloader, criterion, Group_Num, Sample_Num):
    PSNR = 0.
    np.set_printoptions(precision=2)
    torch.set_printoptions(precision=2)
    for i in range(Group_Num):
        print("Current test turn: %d" %(i))
        PSNR += test_deepsc_tx(net, device, testloader, criterion, Sample_Num)
    np.save(f'simulation/experiment_1/PSNR.npy', PSNR/Group_Num)
    print('PSNR : ', torch.tensor(PSNR/Group_Num))
    exit(0)


# Testing the model
def test_deepsc_snr(net, device, dataloader, criterion, sample_num):
    net.eval()
    SNRdBs = np.arange(0, 20, 2) 
    tx_time = 2
    PSNR_Group = np.zeros(len(SNRdBs))
    with torch.no_grad():
        test_loss = 0
        MSE_loss = 0
        for i in range(len(SNRdBs)):
            snr = SNRdBs[i]
            print('Test SNR = %d dB' % (snr) )
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), inputs.to(device)
                for tx in range(tx_time):
                    outputs = net(inputs, snr=snr)                                 
                    loss = criterion(outputs, targets)                              
                    test_loss = loss.item()

                    predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                    target = torch.chunk(targets, chunks=inputs.size(0), dim=0)
                    
                    ######  PSNR/SSIM/etc  ######
                    psnr_vals = calc_psnr(predictions, target)
                    PSNR = torch.mean(torch.tensor(psnr_vals)).item()

                    inputs = outputs[:]
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Mse: %.3f | PSNR: %.2f'
                                % (test_loss, MSE_loss/(batch_idx+1), PSNR))
                PSNR_Group[i] += PSNR
            print(PSNR_Group/(batch_idx+1))
        PSNR_Group /= (batch_idx+1) 
    return PSNR_Group


##  This function is used to test the scheme in different SNRs
def Experiment_2(net, device, testloader, criterion, Group_Num, Sample_Num):
    PSNR = 0.
    np.set_printoptions(precision=2)
    torch.set_printoptions(precision=2)
    for i in range(Group_Num):
        print("Current test turn: %d" %(i))
        PSNR += test_deepsc_snr(net, device, testloader, criterion, Sample_Num)
    np.save(f'simulation/experiment_2/PSNR.npy', PSNR/Group_Num)
    print('PSNR : ', torch.tensor(PSNR/Group_Num))
    exit(0)
