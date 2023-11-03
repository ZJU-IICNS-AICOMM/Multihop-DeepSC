'''Test the model with PyTorch.'''
'''Usage: python3 sim_main.py  --cr 0.3  --resume checkpoint/checkpoint-fn14.pth  --test'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from functools import partial
import torchvision.transforms as transforms
import sys
import os
import argparse

from modules import *
from utils import *
from model import *
from sim_engine import *

def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--test_batch_size', default=100, type=int, help='Testing batchsize')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--model', default='DeepSC', type=str, help='model name')

    parser.add_argument('--output_dir', default='output', help='Path where to save, empty for no saving')

    ## Model parameters
    parser.add_argument('--cr', default=4, type=int, help='Compression ratio for the image')

    ## Test Settings
    parser.add_argument('--test', action='store_true', help='Whether only test the model')
    
    parser.add_argument('--test_number', default='', choices=['Experiment_1','Experiment_2', 'Experiment_3', 'Experiment_4', 'Experiment_5', 'Experiment_6'],
                        type=str, help='Eval number of the experiment')
    
    args = parser.parse_args()
    return args

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_initial(1000)

    # Data
    print('==> Preparing data..')
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10_RES3(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, num_workers=2)

    # Model
    if args.model.startswith('DeepSC'):
        num_filters, compression_ratio = calculate_filters(args.cr)
        print('==> Building model.. DeepSCNet \nThe real compression rate is %.3f || The number of filters is %d' % (compression_ratio, num_filters))
        net = DeepSCNet(filters=num_filters)
        net = net.to(device)
        n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('number of params: {}M'.format(n_parameters / 1e6))
    elif args.model.startswith('ViTSC'):
        print('==> Building model.. ViTSC  \nThe real compression rate is %.3f' % (args.cr))
        net = ViTSCNet( compression_ratio=args.cr, 
                        img_size=32,
                        patch_size=4,
                        encoder_embed_dim=128,
                        encoder_depth=4,
                        encoder_num_heads=6,
                        decoder_embed_dim=128,
                        decoder_depth=2,
                        decoder_num_heads=4,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        net = net.to(device)
        n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('number of params: {}M'.format(n_parameters / 1e6))

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        best_mse = checkpoint['mse']
        start_epoch = checkpoint['epoch'] 
    criterion = nn.MSELoss()
    
    Group_Num =  1         # The total number of times of the experiments
    Sample_Num = 1000         # The total number of tested samples
    if args.test:
        if   args.test_number == 'Experiment_1':
            Experiment_1(net, device, testloader, criterion, Group_Num, Sample_Num)
        elif args.test_number == 'Experiment_2':
            Experiment_2(net, device, testloader, criterion, Group_Num, Sample_Num)
        # elif args.test_number == 'Experiment_3':
        #     Experiment_3(net, device, testloader, criterion, Group_Num, Sample_Num)
        # elif args.test_number == 'Experiment_4':
        #     Experiment_4(net, device, testloader, criterion, Group_Num, Sample_Num)
        # elif args.test_number == 'Experiment_5':
        #     Experiment_5(net, device, testloader, criterion, Group_Num, Sample_Num)
            
if __name__ == '__main__':
    opts = get_args()
    disp_args(opts)
    main(opts)