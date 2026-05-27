from __future__ import print_function

import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = torch.sum(correct[:k].float())
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """
    Sets the learning rate to the initial LR decayed by decay rate every steep step
    根据当前的epoch数和学习率衰减配置，调整优化器的学习率

    Parameters:
    epoch (int): 当前的epoch数
    opt (object): 包含学习率衰减配置的选项对象
    optimizer (object): 优化器实例

    Returns:
    None
    """
    # 计算当前epoch相对于学习率衰减点的位置，以确定学习率的衰减步骤
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        # 根据衰减步骤计算新的学习率
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        # 更新优化器中每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter2:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy_2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []

        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        res.append(correct_1.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass

