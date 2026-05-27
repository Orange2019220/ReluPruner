"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# from models_original import model_dict
# from models_original.util import Embed, ConvReg, LinearEmbed
# from models_original.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

from helper.util import adjust_learning_rate
# from meters import  ScalarMeter, AverageMeter, flush_scalar_meters

from helper.loops import validate as validate_normal
from helper.pretrain import init
from helper.attack import PGD, FGSM
from distiller_zoo import DistillKL


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tiny_imagenet'], help='dataset')


    parser.add_argument('--width_mult_list', '-wml', default=[0.5, 1.0], type=float, help='supporting width multiplier values')
    # model
    # The  resnet50,38,26 matches the models of SASA paper and
    # are written by us. The ResNet50 model is of original CRD paper
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'ODResNet18', 'ODResNet34', 
                                 'ResNet50', 'ResNet18', 'ResNet34',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomODResNet18', 'CustomResNet18', 'CustomODResNet34', 'CustomODvgg16', 'CustomODwrn_22_8'])
    parser.add_argument('--path', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--validate_mode', type=str, default=None, choices=['Normal_Network', 'OD_Network', 'Custom_OD_Network', 'Custom_Network'])

    # attack parameters
    # parser.add_argument('--attack_mode', type=str, default=None, choices=['pgd', 'fgsm'])
    # parser.add_argument('--eps', type=float, default=8/255)
    # parser.add_argument('--steps', type=int, default=7)

    opt = parser.parse_args()

    

    return opt



def load_model(model_path, n_cls, opt):
    print('==> loading model')
    print(opt.model)
    model_name = opt.model
    model = model_dict[model_name](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    if opt.validate_mode == 'Normal_Network' or opt.validate_mode == 'OD_Network':
        return model
    elif opt.validate_mode == 'Custom_OD_Network' or opt.validate_mode == 'Custom_Network':
        return model, torch.load(model_path)['mask_epoch']

def cal_mask_relu(mask_list):
    total_relu = 0
    OD_total_relu = 0

    for i in range(len(mask_list)):
        total_relu += mask_list[i].sum()
        mask_fore, mask_behind = mask_list[i].split(int(mask_list[i].shape[0]/2), 0)
        OD_total_relu += mask_fore.sum()
    print('====================================================================')
    print('Total_relu:', float(total_relu))
    print('The first half channel relu:', float(OD_total_relu))
    print('====================================================================')

def cal_nonmask_relu(model):
    data = torch.randn(2, 3, 32, 32).cuda()
    features = []
    model.eval()
    with torch.no_grad():
        out_t, features = model(data, features, is_feat = False)
    
    counts = 0
    for feature_idx in range(len(features)):
        counts = counts + features[feature_idx].shape[1] * features[feature_idx].shape[2] * features[feature_idx].shape[3]
    print('==========================================')
    print('Total counts of relus:', counts)
            

def main():

    opt = parse_option()
    attacker = None
    criterion_div = DistillKL(1)
#     if opt.attack_mode == 'pgd':
#         attacker = PGD(eps=opt.eps, steps=opt.steps)
#         # attacker = PGD(eps=opt.eps, steps=opt.steps, criterion=criterion_div)
#
#     elif opt.attack_mode == 'fgsm': attacker = FGSM(eps=opt.eps        # attacker = PGD(eps=opt.eps, steps=opt.steps)
# )
#
    if opt.validate_mode == 'Custom_OD_Network':
        opt.od_training = True

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
    elif opt.dataset == 'cifar10':

        train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True, attacker=attacker)
        n_cls = 10
    
    elif(opt.dataset == 'tiny_imagenet'):
        data_transforms = { 'train': transforms.Compose([transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()]),
                            'val'  : transforms.Compose([transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),]) }

        data_dir = '../tiny-imagenet-200/tiny-imagenet-200'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                            for x in ['train', 'val']}
        
        train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batch_size, \
            shuffle=True, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=100, shuffle=True,\
                num_workers=4, pin_memory=True)
        n_cls = 200

    else:
        raise NotImplementedError(opt.dataset)

    # model
    # import pdb; pdb.set_trace()
    if opt.validate_mode == 'Normal_Network' or opt.validate_mode == 'OD_Network':
        model = load_model(opt.path, n_cls, opt)
    elif opt.validate_mode == 'Custom_OD_Network' or opt.validate_mode == 'Custom_Network':
        model, mask_epoch = load_model(opt.path, n_cls, opt)
        
    

    criterion_cls = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion_cls.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    if opt.validate_mode == 'Normal_Network':
        # cal_nonmask_relu(model)
        begin_time = time.time()

        teacher_acc, robust_acc, _, _ = validate_normal(val_loader, model, criterion_cls, opt, attacker=attacker)
        total_time = begin_time - time.time()
        print('==========================================================')
        print('The acc of {} is:{}'.format(opt.model, str(float(teacher_acc))))


    elif opt.validate_mode == 'Custom_Network':
        cal_mask_relu(mask_epoch)
        begin_time = time.time()
        teacher_acc, robust_acc, _, _ = validate_normal(val_loader, model, criterion_cls, opt, mask_epoch, attacker)
        total_time = begin_time-time.time()
        print('==========================================================')
        print('The natural acc of {} is:{}'.format(opt.model, str(float(teacher_acc))))
        print('The total time of {} is:{}'.format(opt.model, str(float(total_time))))




if __name__ == '__main__':
    main()


