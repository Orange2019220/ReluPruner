import copy
import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, optimizer, verbose=True):

        self.verbose = verbose

        self.masks_c = {}
        self.masks_a = {}
        self.modules = []
        # self.original_modules = []
        self.names = []
        self.optimizer = optimizer

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
        # self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None


        #Souvik:SNN: Added this variable to choose if we want to apply the 
        #whole pruning in the CONV layer only or both CONV+FC Linear
        #Default value is false. So, when you wish to prune only CONV
        # make it True.
        self.prune_conv_only = True
       
       
    def init(self, mode='constant', density=0.5):
        self.sparsity = density
       
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks_c: continue
                    #Souvik: added the following part to initialize filter sparsity wise.--yet to add any
                    #self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.masks_c[name][:] = (torch.rand(weight.shape) < density).float().data
                    self.masks_a[name][:] = torch.ones_like(self.masks_c[name][:]) - self.masks_c[name][:]
                    
                    self.baseline_nonzero += weight.numel()*density

            # self.apply_mask()
        
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
                    
                    self.masks[name][:] = (weight != 0.0).float().data
                    self.baseline_nonzero += weight.numel()*density
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
        for name, weight in self.masks_c.items():
            total_size += weight.numel()
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, density*total_size))

    def remove_weight(self, name):
        if name in self.masks_c:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks_c[name].shape, self.masks_c[name].numel()))
            self.masks_c.pop(name)
            self.masks_a.pop(name)
        elif name+'.weight' in self.masks_c:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks_c[name+'.weight'].shape, self.masks_c[name+'.weight'].numel()))
            self.masks_c.pop(name+'.weight')
            self.masks_a.pop(name+'.weight')
        else:
            print('ERROR',name)

    def step(self):
        ###########################################################################
        # Souvik: Here, the author has done the opt.step() 1st and then did apply mask.
        # I think the author never intend to stop gradent flow 1st. They just update the weights with
        # graident, and then make the masking. I think it has similar effect as applying the mask
        # on the gradients and then upadte the weights (like we do in general). 
        ###########################################################################
        self.restore_original_model()
        self.optimizer.step()

        #Update original model with updated train parameters
        self.original_module = copy.deepcopy(self.modules[0]) 

        #Souvik: Following line is added to call the z & u update function before applying the mask to the weights
        
        # self.apply_mask()
        # # for linear the following line decreases the prune_rate linearly, and assigns the updated 
        # # prune rate to prune_rate
        # self.prune_rate_decay.step()
        # self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        # #the minibatch step count is updated by 1
        # self.steps += 1

        # if self.prune_every_k_steps is not None:
        #     if self.steps % self.prune_every_k_steps == 0:
        #         self.truncate_weights()
        #         if self.verbose:
        #             self.print_nonzero_counts()

    def add_module(self, module, density, sparse_init='constant'):
        self.modules.append(module)
        self.original_module = copy.deepcopy(module) 
       
        for name, tensor in module.named_parameters():
            self.names.append(name)
            #self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            self.masks_c[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False)
            self.masks_a[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False)
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
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

    
    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks_c.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks_c[name].shape, np.prod(self.masks_c[name].shape)))
                removed.add(name)
                print('removed layer:{}'.format(name))
                self.masks_c.pop(name)
                self.masks_a.pop(name)

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

    def apply_mask(self, mask_type='nat'):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks_c:
                    if mask_type == 'nat':
                        tensor.data = tensor.data*self.masks_c[name]
                    elif mask_type == 'adv':
                        tensor.data = tensor.data*self.masks_a[name]

    def restore_original_model(self):
        # self.modules.pop()
        # self.modules.append(copy.deepcopy(self.original_module))
        # for org_module, module in zip(self.original_modules, self.modules):
        # for module in self.modules:
        for (org_name, org_tensor), (name, tensor) in zip(self.original_module.named_parameters(), self.modules[0].named_parameters()):
            tensor.data = org_tensor.data

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks_c: continue
                mask_c = self.masks_c[name]
                mask_a = self.masks_a[name]
                # num_nonzeros = (mask != 0).sum().item()
                print('mask_c:', name, (mask_c != 0).sum().item())            
                print('mask_a:', name, (mask_a != 0).sum().item())            