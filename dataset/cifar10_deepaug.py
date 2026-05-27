"""
Reference: https://github.com/amazon-science/normalizer-free-robust-training/blob/main/CRT_TDAT.py
"""


import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
import torch.distributed as dist
import os
import numpy as np    


def cifar_dataset(data_dir, min_crop_scale=0.08, num_classes=10, max_cls=10, attacker=None):
    
    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    test_root = os.path.join(data_dir, 'test')  # this is path to training images folder
    print('Training images loading from %s' % train_root)
    
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
    
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(test_root, transform=test_transform)

    if num_classes > 0 and num_classes < max_cls:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        indices = [i for i, label in enumerate(test_data.targets) if label in selected_classes]
        test_data = Subset(test_data, indices)
        print('Cifar-%d train %d' % (num_classes, len(train_data)))
    
    return train_data, test_data


def cifar_deepaug_dataset(data_dir, min_crop_scale=0.08, num_classes=10, max_cls=10):
    # train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    train_root = data_dir
    print('Training images loading from %s' % train_root)
    
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_data = datasets.ImageFolder(train_root, transform=train_transform)

    if num_classes > 0 and num_classes < max_cls:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        print('Cifar-%d-DeepAug train %d' % (num_classes, len(train_data)))
    
    return train_data


## Texture debias augmentation training dataset
def get_color_distortion(s=0.5):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

def cifar_texture_debias_dataset(data_dir, min_crop_scale=0.64, num_classes=10, max_cls=10):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder

    print('Training images loading from %s (with texture debiased augmentations)' % train_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([get_color_distortion()], p=0.5),
        preprocess]) 

    train_data = datasets.ImageFolder(train_root, transform=train_transform)

    if num_classes > 0 and num_classes < max_cls:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        print('Cifar%d-DeepAug train %d' % (num_classes, len(train_data)))
    
    return train_data


class TriComprehensiveRobustnessDataset(Dataset):
    
    def __init__(self, cifar_data, edsr_data, cae_data, third_domain_data):
        # 30 epochs -> effectively 30 * 3 = 90 epochs
        # In each epoch, 3 * len(imagenet_data) images are trained once.
        self.cifar_data = torch.utils.data.ConcatDataset([cifar_data])
        self.deepaug_data = torch.utils.data.ConcatDataset([edsr_data, cae_data])
        self.third_domain_data = torch.utils.data.ConcatDataset([third_domain_data])
        assert len(self.cifar_data) == len(self.third_domain_data)

    def __getitem__(self, index):
        return self.cifar_data[index], self.deepaug_data[index], self.third_domain_data[index]

    def __len__(self):
        return min(len(self.cifar_data), len(self.deepaug_data))
    

def TriComprehensiveRobustnessDataloader(data_root_path='DeepAugment/', dataset='cifar10', n_cls=10, batch_size=128, num_workers=8, is_instance=False, distributed=False, attacker=None):
     # data loader:
    num_classes = n_cls
    
    train_data, val_data = cifar_dataset(data_dir=os.path.join(data_root_path, dataset), num_classes=num_classes, max_cls=n_cls, attacker=attacker)
    edsr_data = cifar_deepaug_dataset(data_dir=os.path.join(data_root_path, 'EDSR', dataset), num_classes=num_classes, max_cls=n_cls)
    cae_data = cifar_deepaug_dataset(data_dir=os.path.join(data_root_path, 'CAE', dataset), num_classes=num_classes, max_cls=n_cls)
    texture_debias_data = cifar_texture_debias_dataset(data_dir=os.path.join(data_root_path, dataset), num_classes=num_classes, max_cls=n_cls)
    # combine datasets:
    # train_data = ComprehensiveRobustnessDataset(cifar10_data=train_data, edsr_data=edsr_data, cae_data=cae_data)
    train_data = TriComprehensiveRobustnessDataset(cifar_data=train_data, edsr_data=edsr_data, cae_data=cae_data, third_domain_data=texture_debias_data)
    n_data = len(train_data)
    if distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        sampler_val = torch.utils.data.distributed.DistributedSampler(
                val_data, shuffle=False
            )
    else:
        sampler_train = None
        sampler_val = None
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              sampler=sampler_train,
                              shuffle=True if sampler_train is None else False,
                              num_workers=num_workers)

    test_loader = DataLoader(val_data,
                             batch_size=int(batch_size/2),
                             sampler=sampler_val,
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
    



def cifar_c_testloader(corruption, data_dir='./data/', num_classes=10, 
    test_batch_size=100, num_workers=4):
    '''
    Returns:
        test_c_loader: corrupted testing set loader (original cifar10-C)
    CIFAR10-C has 50,000 test images. 
    The first 10,000 images in each .npy are of level 1 severity, and the last 10,000 are of level 5 severity.
    '''

    # # download:
    # url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    # root_dir = data_dir
    # tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    # if not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C.tar')):
    #     download_and_extract_archive(url, root_dir, extract_root=root_dir, md5=tgz_md5)
    # elif not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C')):
    #     extract_archive(os.path.join(root_dir, 'CIFAR-10-C.tar'), to_path=root_dir)

    if num_classes==10:
        dataset = 'cifar10'
        CIFAR = datasets.CIFAR10
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C')
    elif num_classes==100:
        dataset = 'cifar100'
        CIFAR = datasets.CIFAR100
        base_c_path = os.path.join(data_dir, 'CIFAR-100-C')
    else:
        raise Exception('Wrong num_classes %d' % num_classes)
    
    # test set:
    # We use normalization in test transform since the original test loader also contains normalization
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            # 'cifar100': (0.5, 0.5, 0.5),
        }
    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        # 'cifar100': (0.5, 0.5, 0.5),
    }
    mean = mean[dataset]
    std = std[dataset]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_set = CIFAR(data_dir, train=False, transform=test_transform, download=False)
    
    # replace clean data with corrupted data:
    test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))
    test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    print('loader for %s ready' % corruption)

    test_c_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    n_data = len(test_set)
    return test_c_loader, n_data