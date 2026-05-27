# 导入基础库
from __future__ import print_function  # 确保兼容Python2/3
import os
import argparse  # 命令行参数解析
import torch
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.tiny_imagenet import get_tiny_imagenet_dataloaders

# 自定义模块
from models import model_dict  # 模型库

import yaml  # 配置文件解析
import sys


def parse_options(method):
    parser = argparse.ArgumentParser('训练参数')

    parser.add_argument('--print_freq', type=int, default=100, help='训练日志打印频率')


    parser.add_argument('--global_keep_ratio', type=float, default=0.1, help='训练比例')

    parser.add_argument('--depth_bias', type=float, default=0.2, help='深度因子')


    # 随机种子配置
    parser.add_argument("--seed", help="随机种子", default='0')

    parser.add_argument('--pretrain_load', type=bool, default= False, help='是否加载预训练权重')
    # parser.add_argument('--sensitivity', type=str, default='ResNet18_c100_relu50k_sensitivity', help='敏感度配置文件')

    # 数据相关参数
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--t1_epochs', type=int, default=150, help='模型训练轮数')
    parser.add_argument('--init_epochs', type=int, default=30, help='初始训练阶段轮数')

    # 优化器参数
    parser.add_argument('--learning_rate', type=float, default=0.05, help='学习率')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='学习率衰减节点')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减系数')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量参数')


    # 损失权重参数
    parser.add_argument('-r', '--gamma', type=float, default=1, help='分类损失权重')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='知识蒸馏损失权重')
    parser.add_argument('-b', '--beta', type=float, default=1, help='其他损失权重')


    # 数据集配置
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar100', 'cifar10', 'tiny_imagenet', 'imagenet'], help='选择数据集')

    # 模型配置
    parser.add_argument('--model_s', type=str, default='CustomResNet18', help='学生模型架构')
    parser.add_argument('--path_t', type=str, default='ResNet18', help='教师模型路径')

    # KL散度温度参数
    parser.add_argument('--kd_T', type=float, default=4, help='知识蒸馏温度')
    parser.add_argument('--kd_T_adv', type=float, default=None, help='对抗样本温度')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd','attention'], help='蒸馏方法选择')

    parser.add_argument("--local-rank", default=0, type=int)




    opt = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(int(opt.seed))

    # 特殊模型调整学习率
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01


    # 解析学习率衰减节点
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    # 解析教师模型名称
    opt.model_t = get_teacher_name(opt.path_t)

    # 构建模型命名规则
    # relu_count = opt.sensitivity.split('sensitivity')[0].split('_')[-2]
    relu_count = opt.global_keep_ratio

    opt.model_name = 'S_{}_T1_{}_{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, relu_count)
    # opt.model_name += '_OnlyTalyor' # 使用的方法是senet
    opt.model_name +='_'+ method # 使用的方法
    opt.model_name += '_kdT{}'.format(opt.kd_T)     # 添加知识蒸馏参数标识
    opt.model_name += '_batch{}'.format(opt.batch_size) # 添加批量大小标识

    # 路径配置
    opt.model_path = './save/student_model/stage1'
    opt.tb_path = './save/student_tensorboards/stage1'
    # 创建tensorborder保存路径
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
    # 保存路径
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)
    # 日志文件路径
    opt.outfile = os.path.join(opt.save_folder,'results.out')

    # 加载敏感度配置......................................................................................直接加载进来
    # with open('sensitivity_list.yaml', 'r') as f:
    #     sensitivity_yaml = yaml.safe_load(f.read())
    # opt.sensitivity_list = sensitivity_yaml[opt.sensitivity]

    return  opt


def get_teacher_name(model_path):
    """从模型路径解析教师模型名称"""
    segments = model_path.split('/')[-1].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(opt, model_path, n_cls):
    """加载教师模型"""
    print('==> 加载教师模型')
    # model path = save/models/ResNet18_tiny_imagenet/best.pth.tar
    model_t = get_teacher_name(model_path)
    print(model_t)
    model = model_dict[model_t](num_classes=n_cls)
    # 加载预训练权重
    if opt.dataset == 'tiny_imagenet':
        state_dict = torch.load(model_path)['state_dict']
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)  # 允许部分匹配
    else:
        model.load_state_dict(torch.load(model_path)['model'])
    print(model)
    print('==> 加载完成')
    return model


def load_dataset(opt):
    # train_loader, val_loader, n_data, n_cls = False

    if opt.dataset == 'cifar100':
        # CIFAR100数据集加载
        train_loader, val_loader, n_data = get_cifar100_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=True
        )
        n_cls = 100
    elif opt.dataset == 'cifar10':
        # CIFAR10数据集加载
        train_loader, val_loader, n_data = get_cifar10_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            distributed=opt.distributed,
            is_instance=True
        )
        n_cls = 10
    elif opt.dataset == 'tiny_imagenet':
         train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders(batch_size=opt.batch_size,
                                                                             num_workers=opt.num_workers,
                                                                             distributed=opt.distributed,
                                                                             is_instance=True)
         n_cls = 200
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                   num_workers=opt.num_workers,
                                                                   distributed=opt.distributed,
                                                                   is_instance=True)
        n_cls = 1000
    else:
        raise NotImplementedError(opt.dataset)

    return   train_loader, val_loader, n_data ,n_cls


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val_f(self):
        return float(self.val.item()) if isinstance(self.val, torch.Tensor) else float(self.val)

    @property
    def avg_f(self):
        return float(self.avg.item()) if isinstance(self.avg, torch.Tensor) else float(self.avg)


def accuracy(output, target, topk=(1,)):
    """计算精度"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

