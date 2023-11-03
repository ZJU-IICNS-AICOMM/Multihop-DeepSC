'''Train CIFAR10 with PyTorch.'''
'''Usage: python3 deepsc_main.py  --lr 0.0001  --cr 0.3  --resume checkpoint/ckpt_cr4_CNN_P1.pth  --test'''
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from functools import partial
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import *
from model import DeepSCNet,ViTSCNet
from engine import *


def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='Training batchsize')
    parser.add_argument('--test_batch_size', default=100, type=int, help='Testing batchsize')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of taining epochs')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--model', default='DeepSC', type=str, help='model name')
    

    parser.add_argument('--output_dir', default='output', help='Path where to save, empty for no saving')
    parser.add_argument('--save_ckpt', default=True, help='Wether save the checpoint')
    parser.add_argument('--save_freq', default=3, type=int, help='Frequency to save tyhe model')

    ## Model parameters
    parser.add_argument('--cr', default=4, type=float, help='Compression ratio for the image')

    ## Test Settings
    parser.add_argument('--test', action='store_true', help='Whether only test the model')
    
    ## Gen data settings
    parser.add_argument('--gendata', action='store_true', help='Whether only generate the data of GAN model')
    args = parser.parse_args()
    return args


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_initial(1010)
    best_mse = 1000  # best test mse
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)


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
        # best_mse = checkpoint['mse']
        best_mse = np.inf
        start_epoch = checkpoint['epoch']

    criterion = nn.MSELoss()              
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    optimizer =  optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    
    if args.gendata:
        gen_data(net, device, testloader, trainloader, criterion)
        print('Successfully generate data!')
        exit(0)
    
    if args.test:
        mse, psnr = test_deepsc(net, device, testloader, criterion)
        # psnr = 10 * math.log10(255.0**2/(mse))
        print('Performance on current checkpoint is MSE: %.3f , PSNR: %.3f dB' % (mse, psnr))
        exit(0)

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        train_deepsc(epoch, net, device, trainloader, criterion, optimizer)
        scheduler.step()

        if testloader is not None:
            mse,psnr = test_deepsc(net, device, testloader, criterion)
            if args.output_dir and args.save_ckpt:
                if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs:
                    print('Saving model..')
                    save_checkpont(args, net, mse, epoch, best_model=False)

            if mse < best_mse:
                print('Saving best model..')
                save_checkpont(args, net, mse, epoch, best_model=True)
                best_mse = mse
            
if __name__ == '__main__':
    opts = get_args()
    disp_args(opts)
    main(opts)