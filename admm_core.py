from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import shutil
import time

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torchvision import datasets, transforms
from numpy import linalg as LA
from matplotlib import pyplot as plt
from funcs import redistribution_funcs, growth_funcs, prune_funcs

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, \
                            filter_regrowth and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET, Filter.')
    parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: \
                            momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--dense', action='store_true', default=False ,help='Enable dense mode. Default: False.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')


class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule
    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    """Anneals the pruning rate linearly with each step."""
    #################################################################
    # prune_rate decreases linearly to 0 from initial (0.5) linearly, stepwise, where the step
    # is decided by the total minibatch size (minibatch_per_epoch x total epochs)
    #################################################################
    def __init__(self, prune_rate, T_max):
        self.steps = 0
        self.decrement = prune_rate/float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate



class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.
    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.
    Basic usage:
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)
    Removing layers: Layers can be removed individually, by type, or by partial
    match of their name.
      - `mask.remove_weight(name)` requires an exact name of
    a parameter.
      - `mask.remove_weight_partial_name(partial_name=name)` removes all
        parameters that contain the partial name. For example 'conv' would remove all
        layers with 'conv' in their name.
      - `mask.remove_type(type)` removes all layers of a certain type. For example,
        mask.remove_type(torch.nn.BatchNorm2d) removes all 2D batch norm layers.
    """
    #Souvik: changed verbose to true for debug
    def __init__(self, optimizer, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=True, fp16=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'filter_momentum', 'channel_momentum ', 'column_momentum']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        #Souvik: Following gorwth and prune mode is taken by the user 
        print('Prune mode taken by user:', prune_mode)
        print('Growth model taken by user:', growth_mode)

        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose

        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None

        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        #urgent: Souvik: changed prune_every_k_steps from None to 100
        self.prune_every_k_steps = 100
        self.half = fp16
        self.name_to_32bit = {}

        #Souvik:SNN: Added this variable to choose if we want to apply the 
        #whole pruning in the CONV layer only or both CONV+FC Linear
        #Default value is false. So, when you wish to prune only CONV
        # make it True.
        self.prune_conv_only = True
        ###############################################
        #Souvik: variables to support admm based loss function
        ###############################################
        self.ADMM_U = {}
        self.ADMM_Z = {}
        #self.rho = 0.0000000001 # We hard code the rhos
        self.rho = 0.0001 # We hard code the rhos (for cifar100 we change the rho value by a factor of 1000 and change from 0.00000001
        ###############################################

    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True
        print(self.optimizer.param_groups[0]['lr'])

    def init(self, mode='constant', density=0.05):
        self.sparsity = density
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    #Souvik: added the following part to initialize filter sparsity wise.--yet to add any
                    #self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data
                    #Souvik: added the following initialization of the admm related variables
                    self.ADMM_U[name] = torch.zeros(weight.shape).cuda()
                    self.ADMM_Z[name] = torch.Tensor(weight.shape).cuda()
                    #self.ADMM_U[name] = torch.zeros(weight.shape)
                    #self.ADMM_Z[name] = torch.Tensor(weight.shape)
                    ######################################################
                    self.baseline_nonzero += weight.numel()*density

            self.apply_mask()
        
        elif mode == 'resume':
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    #self.masks[name][:] = (weight != 0.0).float().data.cuda()
                    self.masks[name][:] = (weight != 0.0).float().data
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif mode == 'linear':
            # initialization used in sparse evolutionary training
            # scales the number of non-zero weights linearly proportional
            # to the product of all dimensions, that is input*output
            # for fully connected layers, and h*w*in_c*out_c for conv
            # layers.

            total_params = 0
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel()*density

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                #self.masks[name][:] = (torch.rand(weight.shape) < prob).float().data.cuda()
                self.masks[name][:] = (torch.rand(weight.shape) < prob).float().data
            self.apply_mask()

        self.print_nonzero_counts()
        with open("stat_gather.txt",'a') as f:
            f.write(50*"##")
        total_size = 0
        for name, module in self.modules[0].named_modules():
            if hasattr(module, 'weight'):
                # Souvik: The following line is added to avoid Nonetype object search for numel() in case 
                # of BatchNorm1d. This is to avoid error occuring for VGG9.
                if isinstance(module, nn.BatchNorm1d): continue
                else:
                    total_size += module.weight.numel()
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    total_size += module.bias.numel()
        print('Total Model parameters:', total_size)

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, density*total_size))

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')

    def at_end_of_epoch(self):
        self.truncate_weights()
        self.print_nonzero_counts()
        print("prune_rate: ", self.prune_rate)
        if self.verbose:
            self.print_nonzero_counts()

    #Souvik: The Z and U variables are updated in the following function, note this function should be
    #called before the step function, to use the unmasked weights
    def z_u_update(self, args, model, device, train_loader, optimizer, epoch, data, batch_idx):
        if epoch !=1 and batch_idx == 0:
            for i, (name, W) in enumerate(model.named_parameters()):
                #Souvik: Following condition is to make sure that the update happens for only CONV layer weights
                if name not in self.masks: continue
                self.ADMM_Z[name] = W +  self.ADMM_U[name] #Z(k+1) = W(k+1)+U(k)
                self.ADMM_U[name] = W - self.ADMM_Z[name] + self.ADMM_U[name] #U(k+1) = W(k+1)-Z(k+1)+U(k)
                
                

    
    #Souvik: The admm and mixed loss is computed in the following function 
    def append_admm_loss(self, args, model, ce_loss):
        admm_loss = {}
        '''
        sum_u = {}
        sum_z = {}
        sum_w = {}
        '''
        for i, (name, W) in enumerate(model.named_parameters()):
            if name in self.masks:
                admm_loss[name] = 0.5 * self.rho * (torch.norm(W - self.ADMM_Z[name] + self.ADMM_U[name], p=2) ** 2)
                #sum_w[name] = (torch.norm(W, p=2)**2)
                #sum_u[name] = (torch.norm(self.ADMM_U[name], p=2)**2)
                #sum_z[name] = (torch.norm(self.ADMM_Z[name], p=2)**2)
                #Souvik: Experimetal: changed L2 dynamic to L1 dynamic
                #admm_loss[name] = 0.5 * self.rho * (torch.norm(W - self.ADMM_Z[name] + self.ADMM_U[name], p=1) ** 1)
        mixed_loss = 0
        mixed_loss += ce_loss
        for k, v in admm_loss.items():
            mixed_loss += v
        #return ce_loss, admm_loss, mixed_loss, sum_w, sum_u, sum_z
        return ce_loss, admm_loss, mixed_loss


    def step(self):
        ###########################################################################
        # Souvik: Here, the author has done the opt.step() 1st and then did apply mask.
        # I think the author never intend to stop gradent flow 1st. They just update the weights with
        # graident, and then make the masking. I think it has similar effect as applying the mask
        # on the gradients and then upadte the weights (like we do in general). 
        ###########################################################################
        
        self.optimizer.step()
        #Souvik: Following line is added to call the z & u update function before applying the mask to the weights
        
        self.apply_mask()
        # for linear the following line decreases the prune_rate linearly, and assigns the updated 
        # prune rate to prune_rate
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        #the minibatch step count is updated by 1
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                if self.verbose:
                    self.print_nonzero_counts()

    def add_module(self, module, density, sparse_init='constant'):
        self.modules.append(module)
        count_alpha = 0
        count_mean = 0
        count_std = 0
        for name, tensor in module.named_parameters():
            self.names.append(name)
            if 'alpha_w' in name:
                count_alpha = count_alpha + 1
            if 'mean' in name:
                count_mean = count_mean + 1
            if 'std' in name:
                count_std = count_std + 1 
            #self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False)
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        #souvik: added the following 2 lines to remove the alpha_w parameters from pruning list
        if count_alpha > 0:
            self.remove_weight_partial_name('alpha_w')
        if count_mean > 0:
            self.remove_weight_partial_name('mean')
        if count_std > 0:
            self.remove_weight_partial_name('std')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d, verbose=self.verbose)
        '''
        Souvik: The following code to function call is added to remove the weights related to Linear layer
        From the pruning mechanism to limit the pruning to only CONV layers.
        '''
        if (self.prune_conv_only == True):
            self.remove_type(nn.Linear, verbose=self.verbose)

        self.init(mode=sparse_init, density=density)

    def is_at_start_of_pruning(self, name):
        if self.start_name is None: self.start_name = name
        if name == self.start_name: return True
        else: return False

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name+'.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name+'.weight'].shape, self.masks[name+'.weight'].numel()))
            self.masks.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape, np.prod(self.masks[name].shape)))
                removed.add(name)
                print('removed layer:{}'.format(name))
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)
                    #self.remove_weight_partial_name(name, verbose=self.verbose)

    #########################################################
    # The following function applies the mask to the weight tensors,
    # for the layers which are in the name list.
    #########################################################

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        # growing
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        if self.global_prune:
            self.total_removed = self.prune_func(self)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    #Souvik: added following line for debug:
                    #print('Mask is:', mask)

                    # prune
                    new_mask = self.prune_func(self, mask, weight, name)
                    removed = self.name2nonzeros[name] - new_mask.sum().item()
                    self.total_removed += removed
                    self.name2removed[name] = removed
                    self.masks[name][:] = new_mask
        print('Remove newly through pruning: {}'.format(self.name2removed))

        name2regrowth = self.calc_growth_redistribution()
        
        if self.global_growth:
            total_nonzero_new = self.growth_func(self, self.total_removed + self.adjusted_growth)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    # growth
                    new_mask = self.growth_func(self, name, new_mask, math.floor(name2regrowth[name]), weight)
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (prune-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        if self.total_nonzero > 0 and self.verbose:
            print('Nonzero before/after: {0}/{1}. Growth adjustment: {2:.2f}.'.format(
                  self.total_nonzero, total_nonzero_new, self.adjusted_growth))

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}
        self.name2removed = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                # redistribution
                self.name2variance[name] = self.redistribution_func(self, name, weight, mask)

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

        for name in self.name2variance:
            if self.total_variance != 0.0:
                self.name2variance[name] /= self.total_variance
            else:
                print('Total variance was zero!')
                print(self.growth_func)
                print(self.prune_func)
                print(self.redistribution_func)
                print(self.name2variance)

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                # It determines the max regrowth value based on what was pruned and what was zero earlier (added)
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if self.prune_mode == 'global_magnitude':

                    expected_removed = self.baseline_nonzero*self.name2prune_rate[name]
                    if expected_removed == 0.0:
                        name2regrowth[name] = 0.0
                    else:
                        expected_vs_actual = self.total_removed/expected_removed
                        name2regrowth[name] = math.floor(expected_vs_actual*name2regrowth[name])
        #Souvik: added following line for debug
        print('Prune mode: {}, name2regrowth {}'.format(self.prune_mode, name2regrowth))
        return name2regrowth


    '''
                UTILITY
    '''
    # Following function is called by funcs.py get_momentum_redistribution to collect the momentum of the weights.
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()

                if name in self.name2variance:
                    #val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    #print(val)
                    with open('stat_gather.txt', 'a') as f:
                        f.write("layer:{}, earlier:{}, now: {}, density: {}, prune_rate: {}".format(name, self.name2nonzeros[name], num_nonzeros, \
						 num_nonzeros/float(mask.numel()), self.prune_rate))
                        f.write("\n\n")
                else:
                    print(name, num_nonzeros)
        #Souvik: The following two print functions are added for debug
        #print('name2prune_rate:{}\n'.format(self.name2prune_rate))
        #print('Prune rate: {0}\n'.format(self.prune_rate))

		
