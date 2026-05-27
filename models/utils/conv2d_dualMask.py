import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2d_dualMask(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(conv2d_dualMask, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        rand_tensor = torch.rand(self.weight.size())
        self.mask_c = nn.Parameter((rand_tensor >= 0.5) + 0.0, requires_grad=False)
        self.mask_a = nn.Parameter((torch.ones_like(self.weight) - self.mask_c), requires_grad=False)
      
        # self.mask_c = self.mask_c.cuda()
        # self.mask_a = self.mask_a.cuda()


    def forward(self, input, mask_type=None):

        if mask_type == 'nat':
            noise_weight = self.weight * self.mask_c
        elif  mask_type == 'adv':
            noise_weight = self.weight * self.mask_a
        else:
            raise Exception('mask_type is not defined!')

        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output    