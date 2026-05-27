import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import numpy as np
import argparse
from operator import itemgetter
from heapq import nsmallest
import dataset


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ReLUPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.relu_ranks = {}
        self.relu_activations = []
        self.relu_gradients = []

    def forward(self, x):
        self.relu_activations = []
        self.relu_gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        # 注册钩子函数来捕获ReLU的激活和梯度
        activation_index = 0

        # 处理features中的ReLU
        for layer_idx, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.ReLU):
                # 为ReLU激活注册钩子
                x.register_hook(self.compute_relu_rank)
                self.relu_activations.append(x)
                self.activation_to_layer[activation_index] = ('features', layer_idx)
                activation_index += 1

        # 展平特征
        x = x.view(x.size(0), -1)

        # 处理classifier中的ReLU
        for layer_idx, (name, module) in enumerate(self.model.classifier._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.ReLU):
                x.register_hook(self.compute_relu_rank)
                self.relu_activations.append(x)
                self.activation_to_layer[activation_index] = ('classifier', layer_idx)
                activation_index += 1

        return x

    def compute_relu_rank(self, grad):
        """
        计算ReLU的重要性分数
        使用泰勒展开近似: 重要性 ≈ |activation * gradient|
        """
        activation_index = len(self.relu_activations) - self.grad_index - 1
        activation = self.relu_activations[activation_index]

        # 计算泰勒展开的一阶项: activation * gradient
        taylor = activation * grad

        # 对于ReLU，我们关心每个神经元的重要性
        # 计算每个神经元在batch和空间维度上的平均重要性
        if len(taylor.shape) == 4:  # Conv层后的ReLU (batch, channel, height, width)
            taylor = taylor.mean(dim=(0, 2, 3)).data  # 对batch, height, width求平均
        elif len(taylor.shape) == 2:  # FC层后的ReLU (batch, features)
            taylor = taylor.mean(dim=0).data  # 对batch求平均

        # 使用绝对值作为重要性指标
        taylor = torch.abs(taylor)

        if activation_index not in self.relu_ranks:
            self.relu_ranks[activation_index] = torch.FloatTensor(taylor.size()).zero_()
            if args.use_cuda:
                self.relu_ranks[activation_index] = self.relu_ranks[activation_index].cuda()

        self.relu_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_relus(self, num):
        """获取重要性最低的ReLU神经元"""
        data = []
        for activation_idx in sorted(self.relu_ranks.keys()):
            layer_info = self.activation_to_layer[activation_idx]
            for neuron_idx in range(self.relu_ranks[activation_idx].size(0)):
                importance_score = self.relu_ranks[activation_idx][neuron_idx]
                data.append((layer_info, neuron_idx, importance_score))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        """归一化每层的重要性分数"""
        for i in self.relu_ranks:
            v = torch.abs(self.relu_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v) + 1e-8)  # 避免除零
            self.relu_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_relus_to_prune):
        """生成ReLU剪枝计划"""
        relus_to_prune = self.lowest_ranking_relus(num_relus_to_prune)

        # 按层组织要剪枝的ReLU
        relus_to_prune_per_layer = {}
        for (layer_info, neuron_idx, _) in relus_to_prune:
            if layer_info not in relus_to_prune_per_layer:
                relus_to_prune_per_layer[layer_info] = []
            relus_to_prune_per_layer[layer_info].append(neuron_idx)

        # 排序并调整索引（考虑之前已经剪枝的影响）
        for layer_info in relus_to_prune_per_layer:
            relus_to_prune_per_layer[layer_info] = sorted(relus_to_prune_per_layer[layer_info])
            for i in range(len(relus_to_prune_per_layer[layer_info])):
                relus_to_prune_per_layer[layer_info][i] = relus_to_prune_per_layer[layer_info][i] - i

        # 转换为最终的剪枝列表
        final_pruning_plan = []
        for layer_info in relus_to_prune_per_layer:
            for neuron_idx in relus_to_prune_per_layer[layer_info]:
                final_pruning_plan.append((layer_info, neuron_idx))

        return final_pruning_plan


class PruningFineTuner_ReLU:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = ReLUPrunner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
                label = label.cuda()

            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        accuracy = float(correct) / total
        print(f"Accuracy: {accuracy:.4f}")
        self.model.train()
        return accuracy

    def train_batch(self, optimizer, batch, label, rank_relus):
        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input_var = Variable(batch)

        if rank_relus:
            output = self.prunner.forward(input_var)
            loss = self.criterion(output, Variable(label))
            loss.backward()
        else:
            output = self.model(input_var)
            loss = self.criterion(output, Variable(label))
            loss.backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_relus=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_relus)

    def train(self, optimizer=None, epochs=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def get_candidates_to_prune(self, num_relus_to_prune):
        """获取要剪枝的ReLU候选"""
        self.prunner.reset()
        self.train_epoch(rank_relus=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_pruning_plan(num_relus_to_prune)

    def total_num_relus(self):
        """计算总的ReLU神经元数量"""
        total_relus = 0

        # 计算features中的ReLU
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.ReLU):
                # 需要运行一次前向传播来获取形状
                # 这里简化处理，实际应该根据前一层的输出形状计算
                pass

        # 计算classifier中的ReLU
        for name, module in self.model.classifier._modules.items():
            if isinstance(module, torch.nn.ReLU):
                # 对于全连接层后的ReLU，神经元数等于前一层的输出特征数
                pass

        return total_relus

    def prune_relu_neurons(self, pruning_plan):
        """
        执行ReLU神经元剪枝
        注意：这是一个简化的实现示例
        实际实现需要更复杂的网络结构修改
        """
        print("执行ReLU剪枝...")

        # 创建掩码来"软剪枝"ReLU神经元
        self.relu_masks = {}

        for (layer_info, neuron_idx) in pruning_plan:
            layer_type, layer_idx = layer_info
            key = f"{layer_type}_{layer_idx}"

            if key not in self.relu_masks:
                # 初始化掩码（这里需要根据实际层的大小来设置）
                if layer_type == 'features':
                    # 卷积层后的ReLU，需要获取通道数
                    pass
                elif layer_type == 'classifier':
                    # 全连接层后的ReLU
                    pass

            # 将对应神经元的掩码设为0
            # self.relu_masks[key][neuron_idx] = 0

        print(f"剪枝了 {len(pruning_plan)} 个ReLU神经元")

    def apply_relu_masks(self, x, layer_type, layer_idx):
        """应用ReLU掩码"""
        key = f"{layer_type}_{layer_idx}"
        if key in self.relu_masks:
            return x * self.relu_masks[key]
        return x

    def prune(self):
        """执行ReLU剪枝的主要流程"""
        print("开始ReLU剪枝...")

        # 获取剪枝前的准确率
        print("剪枝前的准确率:")
        self.test()

        # 设置所有层为可训练
        for param in self.model.parameters():
            param.requires_grad = True

        # 剪枝参数
        num_relus_to_prune_per_iteration = 100  # 每次迭代剪枝的ReLU数量
        iterations = 5  # 剪枝迭代次数

        print(f"计划进行 {iterations} 次剪枝迭代")

        for iteration in range(iterations):
            print(f"\n=== 剪枝迭代 {iteration + 1}/{iterations} ===")

            print("计算ReLU重要性分数...")
            prune_targets = self.get_candidates_to_prune(num_relus_to_prune_per_iteration)

            print(f"将要剪枝 {len(prune_targets)} 个ReLU神经元")

            # 执行剪枝
            self.prune_relu_neurons(prune_targets)

            # 测试剪枝后的性能
            print("剪枝后的准确率:")
            self.test()

            # 微调恢复性能
            print("微调以恢复性能...")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epochs=5)

        print("\n最终微调...")
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.train(optimizer, epochs=10)

        print("ReLU剪枝完成!")
        torch.save(self.model.state_dict(), "model_relu_pruned")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args


if __name__ == '__main__':
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model()
    elif args.prune:
        model = torch.load("model", map_location=lambda storage, loc: storage)

    if args.use_cuda:
        model = model.cuda()

    fine_tuner = PruningFineTuner_ReLU(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epochs=10)
        torch.save(model, "model")
    elif args.prune:
        fine_tuner.prune()