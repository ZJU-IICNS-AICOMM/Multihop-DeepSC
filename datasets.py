import os
import cv2
import glob
import ipdb
import torch
import random
import pickle
import os.path
import numpy as np

from PIL import Image
from utils import np_to_torch
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def crop_cv2(img, patch):
    height, width, _ = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    return img[start_x:start_x + patch, start_y:start_y + patch]




class ImageNet(data.Dataset):
    def __init__(self, fns, mode, args):
        self.fns = fns
        self.mode = mode
        self.args = args
        self.get_image_list()

    def get_image_list(self):
        random.Random(4).shuffle(self.fns)
        num_images = len(self.fns)
        train_size = int(num_images // 1.25)
        eval_size = int(num_images // 10)
        if self.mode == 'TRAIN':
            self.fns = self.fns[:train_size]
        elif self.mode == 'VALIDATE':
            self.fns = self.fns[train_size:train_size+eval_size]
        elif self.mode == 'EVALUATE':
            self.fns = self.fns[train_size+eval_size:train_size+2*eval_size]
        print('Number of {} images loaded: {}'.format(self.mode, len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        image = cv2.imread('datasets/' + image_fn)

        height, width, _ = image.shape
        if height < 128 or width < 128:
            return None, image_fn

        image = crop_cv2(image, self.args.crop)
        image = np_to_torch(image)
        image = image / 255.0
        return image, image_fn

    def __len__(self):
        return len(self.fns)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--crop', default=128, type=int,
                            help='crop size of images')
        return parser


class CIFAR10(data.Dataset):
    def __init__(self, path, mode):
        train_data = np.empty((50000, 32, 32, 3), dtype=np.uint8)
        train_labels = np.empty(50000, dtype=np.uint8)
        for i in range(0, 5):
            data_train = unpickle(os.path.join(path, 'data_batch_{}'.format(i+1)))
            train_data[i*10000:(i+1)*10000] = data_train[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            train_labels[i * 10000:(i + 1) * 10000] = data_train[b'labels']
        self.train = train_data, train_labels
        data_test = unpickle(os.path.join(path, 'test_batch'))
        test_set = data_test[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)), data_test[b'labels']
        self.test = (test_set[0][:10000], test_set[1][:10000])
        self.valid = (test_set[0][:10000], test_set[1][:10000])

        if mode == 'train':
            self.dataset = self.train
        elif mode == 'validate':
            self.dataset = self.valid
        else:
            self.dataset = self.test

    def __getitem__(self, index):
        img, label = self.dataset[0][index], self.dataset[1][index]
        img = np_to_torch(img) / 255.
        return img, int(label)

    def __len__(self):
        return len(self.dataset[0])


class Kodak(data.Dataset):
    def __init__(self, path, args):
        self.path = path
        self.get_image_list()

    def get_image_list(self):
        self.fns = []
        for fn in glob.iglob(self.path + '/*.png', recursive=True):
            self.fns.append(fn)
        print('Number of images loaded: {}'.format(len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        image = cv2.imread(image_fn)

        image = np_to_torch(image)
        image = image / 255.0
        return image, image_fn

    def __len__(self):
        return len(self.fns)




class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class GANDataLoader(object):
    r""" PyTorch DataLoader for GAN loader.
    """

    def __init__(self, root, batch_size, test_batch_size, num_workers, pin_memory):
        assert os.path.isdir(root)
        tx_time = 6
        snr = 18
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_data_raw_train = os.path.join(root, f'data_Txtime_{tx_time:02d}/data_raw_snr_{snr}dB_train.npy')
        dir_data_rec_train = os.path.join(root, f'data_Txtime_{tx_time:02d}/data_rec_snr_{snr}dB_train.npy')
        dir_label_train =    os.path.join(root, f'data_Txtime_{tx_time:02d}/label_snr_{snr}dB_train.npy')
            
            
        dir_data_raw_test = os.path.join(root, f'data_Txtime_{tx_time:02d}/data_raw_snr_{snr}dB_test.npy')
        dir_data_rec_test = os.path.join(root, f'data_Txtime_{tx_time:02d}/data_rec_snr_{snr}dB_test.npy')
        dir_label_test =    os.path.join(root, f'data_Txtime_{tx_time:02d}/label_snr_{snr}dB_test.npy')
        
        

        # Training data loading
        data_raw_train = np.load(dir_data_raw_train)
        data_rec_train = np.load(dir_data_rec_train)
        label_train = np.load(dir_label_train)
        
        data_raw_train = torch.tensor(data_raw_train, dtype=torch.float32)
        data_rec_train = torch.tensor(data_rec_train, dtype=torch.float32)
        label_train = torch.tensor(label_train, dtype=torch.float32)
        

        # Testing data loading
        data_raw_test = np.load(dir_data_raw_test)
        data_rec_test = np.load(dir_data_rec_test)
        label_test = np.load(dir_label_test)
        data_raw_test = torch.tensor(data_raw_test, dtype=torch.float32)
        data_rec_test = torch.tensor(data_rec_test, dtype=torch.float32)
        label_test = torch.tensor(label_test, dtype=torch.float32)
        
    
        self.test_dataset = TensorDataset(data_rec_test, data_raw_test)
        print(len(self.test_dataset))
        self.train_dataset = TensorDataset(data_rec_train, data_raw_train)
        print(len(self.train_dataset))

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.test_batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)

            test_loader = PreFetcher(test_loader)

        return train_loader, test_loader
