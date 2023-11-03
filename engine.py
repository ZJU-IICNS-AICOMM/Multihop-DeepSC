import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from einops import rearrange
import torchvision.transforms as transforms
import numpy as np
import os
import math
import argparse
from tqdm import tqdm
from utils import *
  
# Training DeepSC in a normal manner
# def train_deepsc(epoch, net, device, dataloader, criterion, optimizer, model='ViTSC'):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     MSE_loss = 0
#     snr = 14
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         # snr = np.random.choice(np.arange(0,19,2))
#         inputs, targets = inputs.to(device), inputs.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs,snr=snr)

#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
        
#         transmitted_image = torch.tensor((inputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
#         received_image = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
#         MSE = criterion(transmitted_image, received_image)
#         MSE_loss += MSE.item()

#         PSNR = 10 * math.log10(255.0**2/(MSE_loss/(batch_idx+1)))
#         progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f'
#                      % (train_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), PSNR))

# def test_deepsc(net, device, dataloader, criterion):
#     net.eval()
#     test_loss = 0
#     MSE_loss = 0
#     psnr_list = []
#     psnr = 0
#     snr = 10
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), inputs.to(device)
#             outputs = net(inputs, snr=snr)                                 
#             loss = criterion(outputs, targets)                     
#             test_loss += loss.item()
#             MSE_loss += test_loss
#             ######  Predictions  ######
#             predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
            
#             # target = torch.ones_like(inputs)
#             target = torch.chunk(inputs, chunks=inputs.size(0), dim=0)
#             ######  PSNR/SSIM/etc  ######
            
#             psnr_vals = calc_psnr(predictions, target)
#             psnr_list.extend(psnr_vals)
#             psnr += torch.mean(torch.tensor(psnr_vals)).item()
#             progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
#                         % (test_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
#     MSE = MSE_loss/(batch_idx+1)
#     psnr_list = torch.tensor(psnr_list)
#     PSNR = torch.mean(psnr_list).item()
#     return MSE, PSNR


# Training DJSCC in a recursive manner
def train_deepsc(epoch, net, device, dataloader, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    MSE_loss = 0
    snr = 8
    tx_time = 4
    gamma = 1.15
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), inputs.to(device)
        loss = 0
        # snr = np.random.choice(np.arange(0,19,2))
        optimizer.zero_grad()
        for tx in range(tx_time):
            outputs = net(inputs,snr=snr)
            loss += criterion(outputs, targets) * gamma**(tx_time-tx)
            if tx<(tx_time-1):
                inputs = outputs[:]
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        transmitted_image = torch.tensor((targets*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        received_image = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        
        MSE = criterion(transmitted_image, received_image)
        MSE_loss += MSE.item()

        PSNR = 10 * math.log10(255.0**2/(MSE_loss/(batch_idx+1)))
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Final Mse: %.3e | Final PSNR: %.2f'
                    % (train_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), PSNR))



def test_deepsc(net, device, dataloader, criterion):
    net.eval()
    test_loss = 0
    MSE_loss = 0
    psnr_list = []
    psnr = 0
    snr = 8
    tx_time = 4
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), inputs.to(device)
            for tx in range(tx_time):
                outputs = net(inputs, snr=snr)                                 
                loss = criterion(outputs, targets)                     
                test_loss += loss.item()
                MSE_loss += test_loss
                ######  Predictions  ######
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                
                # target = torch.ones_like(inputs)
                target = torch.chunk(targets, chunks=inputs.size(0), dim=0)
                ######  PSNR/SSIM/etc  ######
                inputs = outputs[:]
            psnr_vals = calc_psnr(predictions, target)
            psnr_list.extend(psnr_vals)
            psnr += torch.mean(torch.tensor(psnr_vals)).item()
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
                        % (test_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
    MSE = MSE_loss/(batch_idx+1)
    psnr_list = torch.tensor(psnr_list)
    PSNR = torch.mean(psnr_list).item()
    return MSE, PSNR



def save_img(source, target, decoded, batch_size=4, epoch=10):
    source = (source*255).detach().cpu().numpy().astype(int).clip(0,255)
    target = (target*255).detach().cpu().numpy().astype(int).clip(0,255)
    decoded= (decoded*255).detach().cpu().numpy().astype(int).clip(0,255)
    import imageio
    def merge_images_two(sources, targets, decoded):
        _, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))

        merged = np.zeros([3, row * h, row * w * 2])
        print(merged.shape)
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)
    
    def merge_images_thr(sources, targets, decoded):
        _, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([3, row * h, row * w * 3])
        for idx, (s, t, de) in enumerate(zip(sources, targets, decoded)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 3) * h:(j * 3 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 3 + 1) * h:(j * 3 + 2) * h] = t
            merged[:, i * h:(i + 1) * h, (j * 3 + 2) * h:(j * 3 + 3) * h] = de
        return merged.transpose(1, 2, 0)
    
    if not os.path.exists('imgs_compare'):
        os.makedirs('imgs_compare')
    path = os.path.join('imgs_compare', f'epoch-{epoch:03d}.jpg')
    merged = merge_images_thr(source, target, decoded)
    imageio.imwrite(path, merged)
      