import torch
import torch.nn as nn

# class DualBN2d(nn.Module):
#     '''
#     Element wise Dual BN, efficient implementation. (Boolean tensor indexing is very slow!)
#     '''
#     def __init__(self, num_features):
#         '''
#         Args:
#             num_features: int. Number of channels
#         '''
#         super(DualBN2d, self).__init__()
#         self.BN_c = nn.BatchNorm2d(num_features)
#         self.BN_a = nn.BatchNorm2d(num_features)

#     def forward(self, _input, idx):
#         '''
#         Args:
#             _input: Tensor. size=(N,C,H,W)
#             idx: int. _input[0:idx,...] -> _lambda=0 -> BN_c; _input[idx:,...] -> _lambda!=0 -> BN_a
        
#         Returns:
#             _output: Tensor. size=(N,C,H,W)
#         '''
#         if idx == 0:
#             _output = self.BN_a(_input)
#         elif idx == _input.size()[0]:
#             _output = self.BN_c(_input)
#         else:
#             _output_c = self.BN_c(_input[0:idx,...]) # BN cannot take tensor with N=0
#             _output_a = self.BN_a(_input[idx:,...])
#             _output = torch.cat([_output_c, _output_a], dim=0)
#         return _output

class DualBN2d(nn.Module):
    '''
    Element wise Dual BN, efficient implementation. (Boolean tensor indexing is very slow!)
    '''
    def __init__(self, num_features):
        '''
        Args:
            num_features: int. Number of channels
        '''
        super(DualBN2d, self).__init__()
        self.BN_c = nn.BatchNorm2d(num_features)
        self.BN_a = nn.BatchNorm2d(num_features)

    def forward(self, _input, idx):
        '''
        Args:
            _input: Tensor. size=(N,C,H,W)
            idx: int. _input[0:idx,...] -> _lambda=0 -> BN_c; _input[idx:,...] -> _lambda!=0 -> BN_a
        
        Returns:
            _output: Tensor. size=(N,C,H,W)
        '''
        if idx == 0:
            _output = self.BN_c(_input)
        elif idx == 1:
            _output = self.BN_a(_input)
        else:
            assert (idx==0 or idx==1), "idx not in (0,1)"
        return _output
    
class TripleBN2d(nn.Module):
    '''
    Element wise Dual BN, efficient implementation. (Boolean tensor indexing is very slow!)
    '''
    def __init__(self, num_features):
        '''
        Args:
            num_features: int. Number of channels
        '''
        super(TripleBN2d, self).__init__()
        self.BN_c = nn.BatchNorm2d(num_features)
        self.BN_r = nn.BatchNorm2d(num_features)
        self.BN_a = nn.BatchNorm2d(num_features)

    def forward(self, _input, idx):
        '''
        Args:
            _input: Tensor. size=(N,C,H,W)
            idx: int. _input[0:idx,...] -> _lambda=0 -> BN_c; _input[idx:,...] -> _lambda!=0 -> BN_a
        
        Returns:
            _output: Tensor. size=(N,C,H,W)
        '''
        if idx == 0:
            _output = self.BN_c(_input)
        elif idx == 1:
            _output = self.BN_r(_input)
        elif idx == 2:
            _output = self.BN_a(_input)
        else:
            assert (idx==0 or idx==1 or idx==2), "idx not in (0,1,2)"
        return _output