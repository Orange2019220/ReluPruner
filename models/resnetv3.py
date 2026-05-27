'''
Written and edited by: 
======================
Souvik Kundu

Description:
=========== 
Following resnet models are written from the            
resnet.py provided by Sairam. These models also 
have the necessary feature  information outputs 
necessary for various distillation techniques.
the code uses ptflops to count FLOPs and PARAMs.
Please do: "pip install ptflops" to install the package. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

model_list = ['ResNet', 'resnet26', 'resnet38', 'resnet50']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last= is_last
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        preact = out
        out = F.relu(out) #don't use self.relu, it somehow impacts the value of preact
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        print('Out shape after Conv1:', out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        print('Out shape after Conv2:', out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        print('Out shape after Conv3:', out.shape)
        #print(out.min().item())
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        preact = out
        #print(preact.min().item(), self.is_last)
        out = F.relu(out) #don't use self.relu, it somehow impacts the value of preact
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.num_classes = num_classes

        # For Imagenet
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        # For CIFAR-10 or CIFAR-100
        if self.num_classes < 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)          

        self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if blocks ==  1:
            last = True
        else:
            last = False
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, is_last=last))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, is_last=(i==blocks-1), dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        f0 = x
        # Reduce image size by 2 in the stem
        # only for Imagenet
        if self.num_classes == 1000:
            x = self.maxpool(x)
        #print(x.shape)
        x, f1_pre = self.layer1(x)
        f1 = x
        #print(f1)
        #print(f1_pre)
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x
        x, f4_pre = self.layer4(x)
        f4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.fc(x)
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x]
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        '''


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet26(**kwargs):
    r"""ResNet-26 model 
    """
    return _resnet(Bottleneck, [1, 2, 4, 1], **kwargs)


def resnet38(**kwargs):
    r"""ResNet-38 model 
    """
    return _resnet(Bottleneck, [2, 3, 5, 2], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

# net = resnet26(num_classes=10)
# x = torch.randn(2, 3, 32, 32)
# feats, logit = net(x, is_feat=True, preact=True)
# flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
# print('Flops:  ' + flops)
# print('Params: ' + params)
#for f in feats:
#    print(f.shape, f.min().item())
#print(logit.shape)
#
#for m in net.get_bn_before_relu():
#    if isinstance(m, nn.BatchNorm2d):
#        print('pass')
#    else:
#        print('warning')
