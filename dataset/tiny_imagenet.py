'''
Tiny-ImageNet:
Download by wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
Run python create_tin_val_folder.py to construct the validation set. 

Tiny-ImageNet-C:
Download by wget https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1
Run python dataloaders/fix_tin_c.py to remove the redundant images in TIN-C.

Tiny-ImageNet-V2:
Download ImageNet-V2 from http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
Run python dataloaders/construct_tin_v2.py to select 200-classes from the full ImageNet-V2 dataset.

https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py
'''

import torch
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np 
import os, shutil
import torch.distributed as dist
import torch
from .augmix import AugMixDataset


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    

def get_tiny_imagenet_dataloaders_augmix(batch_size=100, num_workers=8, is_instance=False, attacker=None, distributed=False):

    data_dir = '/data1/tiny-imagenet-200/'
    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    validation_root = os.path.join(data_dir, 'val')  # this is path to validation images folder
    print('Training images loading from %s' % train_root)
    print('Validation images loading from %s' % validation_root)

    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    # if attacker:
    #     test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])
    # else:
    test_transform = preprocess
    if is_instance:
        train_set = ImageFolderInstance(train_root, transform=train_transform)
        n_data = len(train_set)
    else:
       train_set = datasets.ImageFolder(train_root, transform=train_transform)

    test_set = datasets.ImageFolder(validation_root, transform=test_transform)
    
    train_set_augmix = AugMixDataset(train_set, preprocess, is_instance=is_instance)

    if not attacker:
        train_set = train_set_augmix
    else:
        train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        if is_instance:
            train_set = ImageFolderInstance(train_root, transform=train_transform)
            n_data = len(train_set)
        else:
            train_set = datasets.ImageFolder(train_root, transform=train_transform)
        train_set = ComprehensiveRobustnessDataset(tin_data=train_set, tin_data_augmix=train_set_augmix)

    if distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
                train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        sampler_val = torch.utils.data.distributed.DistributedSampler(
                test_set, shuffle=False
            )
    else:
        sampler_train = None
        sampler_val = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=sampler_train,
                              shuffle=True if sampler_train is None else False,
                              num_workers=num_workers)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             sampler=sampler_val,
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def get_tiny_imagenet_dataloaders(batch_size=100, num_workers=8, is_instance=False, attacker=None, distributed=False):
    
    data_dir = './data/tiny-imagenet-200/'
    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    validation_root = os.path.join(data_dir, 'val')  # this is path to validation images folder
    print('Training images loading from %s' % train_root)
    print('Validation images loading from %s' % validation_root)

    if attacker:
        train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
    if is_instance:
        train_set = ImageFolderInstance(train_root, transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(train_root, transform=train_transform)

    test_set = datasets.ImageFolder(validation_root, transform=test_transform)

    if distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
                train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        sampler_val = torch.utils.data.distributed.DistributedSampler(
                test_set, shuffle=False
            )
    else:
        sampler_train = None
        sampler_val = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True if sampler_train is None else False,
                              sampler=sampler_train,
                              num_workers=num_workers)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             sampler=sampler_val,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def tiny_imagenet_c_testloader(data_dir, corruption, severity, 
    test_batch_size=1000, num_workers=4):

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_root = os.path.join(data_dir, corruption, str(severity))
    test_c_data = datasets.ImageFolder(test_root,transform=test_transform)
    test_c_loader = DataLoader(test_c_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader        
    

class ComprehensiveRobustnessDataset(Dataset):
    def __init__(self, tin_data, tin_data_augmix):
        # 30 epochs -> effectively 30 * 3 = 90 epochs
        # In each epoch, 3 * len(imagenet_data) images are trained once.
        self.tin_data = torch.utils.data.ConcatDataset([tin_data])
        self.tin_data_augmix = torch.utils.data.ConcatDataset([tin_data_augmix])
        assert len(self.tin_data) == len(self.tin_data_augmix)

    def __getitem__(self, index):
        return self.tin_data[index], self.tin_data_augmix[index]

    def __len__(self):
        return min(len(self.tin_data), len(self.tin_data_augmix))