import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

"""
ModifiedVGG16Model类：
构建基于预训练VGG16的修改模型，用于二分类任务。
特征提取层冻结参数，分类层替换为适配新任务的全连接层。
"""
class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        # 加载预训练VGG16模型，保留特征提取层
        model = models.vgg16(pretrained=True)
        self.features = model.features

        # 冻结特征层参数
        for param in self.features.parameters():
            param.requires_grad = False

        # 自定义分类层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        # 前向传播：特征提取 -> 展平 -> 分类
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""
FilterPrunner类：
实现基于梯度泰勒展开的过滤器重要性评估与剪枝计划生成。
主要功能包括：
- 记录卷积层激活值与梯度
- 计算过滤器重要性评分
- 生成最小重要性过滤器列表
"""
class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # 重置过滤器重要性评分记录
        self.filter_ranks = {}

    def forward(self, x):
        # 前向传播并记录卷积层激活值及梯度
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        # 遍历特征层，注册梯度钩子
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        # 计算过滤器泰勒展开重要性评分
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # 计算每个过滤器的平均重要性评分
        taylor = taylor.mean(dim=(0, 2, 3)).data

        # 初始化或累加评分
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        # 获取最低重要性评分的过滤器列表
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        # 对每层的评分进行L2归一化
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        """
        生成剪枝计划并处理索引偏移：
        1. 按重要性排序选择待剪枝过滤器
        2. 处理同层连续剪枝导致的索引变化
        """
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # 统计各层剪枝数量并调整索引
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        # 重组剪枝计划
        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

"""
PrunningFineTuner_VGG16类：
实现剪枝与微调的完整流程管理：
- 数据加载
- 模型训练
- 剪枝操作
- 准确率验证
"""
class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        # 初始化数据加载器、模型和损失函数
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        # 模型验证阶段：计算测试集准确率
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer = None, epoches=10):
        # 主训练流程：执行指定轮数的微调
        if optimizer is None:
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")


    def train_batch(self, optimizer, batch, label, rank_filters):
        # 单批次训练逻辑
        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        # 单轮训练：遍历完整数据集
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        # 获取剪枝候选过滤器的核心流程
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        # 统计特征层总过滤器数量
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        """
        完整剪枝流程：
        1. 评估初始准确率
        2. 迭代剪枝与微调：
           - 计算剪枝候选
           - 执行剪枝操作
           - 微调恢复精度
        3. 最终模型保存
        """
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print("Number of prunning iterations to reduce 67% filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)
        torch.save(model.state_dict(), "model_prunned")

def get_args():
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    # 主程序入口：根据参数执行训练或剪枝
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model()
    elif args.prune:
        model = torch.load("model", map_location=lambda storage, loc: storage)

    if args.use_cuda:
        model = model.cuda()

    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=10)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()
