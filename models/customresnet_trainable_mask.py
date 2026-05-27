'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchextractor as tx
import math
from einops import rearrange, reduce, repeat
# class feature_afterrelu:
#     feature_list = []
#     mask = None

class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):

        self.fea = fea_out

def get_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


def maskedrelu(x, mask):
    assert x.shape == mask[0].shape
    mask_inv = torch.ones_like(mask[0]) - mask[0]
    out = x * mask[0]
    out = F.relu(out)
    out = out + x * mask_inv
    mask.pop(0)
    return out, mask

def dualmaskedrelu(x, mask, mask_adv, idx):
    assert x.shape == mask[0].shape
    assert x.shape == mask_adv[0].shape
    mask_inv = torch.ones_like(mask[0]) - mask[0]
    mask_adv_inv = torch.ones_like(mask_adv[0]) - mask_adv[0]

    if idx == 0:
        out = x * mask_adv[0]
        out = F.relu(out)
        out = out + x * mask_adv_inv
    elif idx == x.size()[0]:
        out = x * mask[0]
        out = F.relu(out)
        out = out + x * mask_inv
    else:
        _out1 = x[0:idx,...] * mask[0][0:idx,...] 
        _out2 = x[idx:,...] * mask_adv[0][idx:,...]
        out = torch.cat([_out1, _out2], dim=0)
        out = F.relu(out)
        out = out + x * torch.cat([mask_inv[0:idx,...], mask_adv_inv[idx:,...]], dim=0)
    mask.pop(0)
    mask_adv.pop(0)
    return out, mask, mask_adv

class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    

class MaskedReLU(nn.Module):
    def __init__(self, sparsity):
        super(MaskedReLU, self).__init__()
        self.sparsity = sparsity
        # initialize the scores
        self.scores = self.register_parameter('scores', None)
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # # NOTE: initialize the weights like this.
        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # # NOTE: turn the gradient on the weights off
        # self.weight.requires_grad = False

    def init_scores(self, input):
        self.scores = nn.Parameter(input.new(input.size()[1:]).normal_(0, 1))  
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, input):
        if self.scores is None:
            self.init_scores(input)
            # self.scores = nn.Parameter(torch.Tensor(input.size()[1:]))
            # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        mask = GetSubnet.apply(self.scores.abs(), self.sparsity)
        # assert input.shape == mask.shape
        mask_inv = torch.ones_like(mask) - mask
        out = input * mask
        out = F.relu(out)
        out = out + input * mask_inv
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, feature_id=0, sensitivity_list=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = MaskedReLU(sparsity=sensitivity_list[feature_id])
        feature_id += 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = MaskedReLU(sparsity=sensitivity_list[feature_id])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, input):
        x, features = input
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # mask = torch.ones_like(out)
        out = self.relu1(out)
        features.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # mask = torch.ones_like(out)
        out = self.relu2(out)
        features.append(out)

        return (out, features)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sensitivity_list=None, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.sensitivity_list = sensitivity_list
        self.feature_id = 0
        self.in_planes = 64
        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = MaskedReLU(sparsity=self.sensitivity_list[self.feature_id])
        self.feature_id += 1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]
    
    def get_mask_list(self):
        mask_list = []
        i = 0
        for m in self.modules():
            if isinstance(m, MaskedReLU):
                mask = GetSubnet.apply(m.scores, self.sensitivity_list[i])
                mask_list.append(mask)
                i+=1
        assert len(self.sensitivity_list) == len(mask_list)
        return mask_list

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, self.feature_id, self.sensitivity_list))
            self.in_planes = planes * block.expansion
            self.feature_id += 2
        return nn.Sequential(*layers)

    def forward(self, x, features, is_feat):
        # out = F.relu(self.bn1(self.conv1(x)))

        out  = self.bn1(self.conv1(x))

        # if mask == None:
        #     print('===========================')
        #     print('Careful!!!! your mask now is None!')
        #     mask = torch.ones_like(out)
        
        out = self.relu1(out)
        f0 = out
        features.append(out)
        if x.shape[-1] == 224: #imagenet
            out = self.maxpool(out)     
        out, features = self.layer1((out, features))
        f1 = out
        # print(layer1.)
        out, features = self.layer2((out, features))
        f2 = out
        out, features = self.layer3((out, features))
        f3 = out
        out, features = self.layer4((out, features))
        f4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out, features


def CustomResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def CustomResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

