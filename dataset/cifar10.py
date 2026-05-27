from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.distributed as dist
import torch
from .augmix import AugMixDataset
"""
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
}t
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar10_dataloaders(batch_size=128, num_workers=8, is_instance=False, attacker=None, distributed=False):
    """
    cifar 10
    """
    data_folder = get_data_folder()

    if attacker:
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if is_instance:
        train_set = CIFAR10Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR10(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    test_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
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


class CIFAR10InstanceSample(datasets.CIFAR10):
    """
    CIFAR10Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        if self.train:
            num_samples = len(self.train_data)
            label = self.train_labels
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar10_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 10
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = CIFAR10InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data


def get_cifar10_dataloaders_augmix(batch_size=128, num_workers=8, is_instance=False, attacker=None, distributed=False):
    """
    cifar 10
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if attacker:
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    else:
        test_transform = preprocess

    if is_instance:
        train_set = CIFAR10Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR10(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    test_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    
    train_set_augmix = AugMixDataset(train_set, preprocess, is_instance=is_instance)

    if not attacker:
        train_set = train_set_augmix
    else:
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        if is_instance:
            train_set = CIFAR10Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
            n_data = len(train_set)
        else:
            train_set = datasets.CIFAR10(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
        train_set = ComprehensiveRobustnessDataset(cifar_data=train_set, cifar_data_augmix=train_set_augmix)

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
    

class ComprehensiveRobustnessDataset(Dataset):
    def __init__(self, cifar_data, cifar_data_augmix):
        # 30 epochs -> effectively 30 * 3 = 90 epochs
        # In each epoch, 3 * len(imagenet_data) images are trained once.
        self.cifar_data = torch.utils.data.ConcatDataset([cifar_data])
        self.cifar_data_augmix = torch.utils.data.ConcatDataset([cifar_data_augmix])
        assert len(self.cifar_data) == len(self.cifar_data_augmix)

    def __getitem__(self, index):
        return self.cifar_data[index], self.cifar_data_augmix[index]

    def __len__(self):
        return min(len(self.cifar_data), len(self.cifar_data_augmix))
    