import torch
import math

#Souvik: added the following two to deal with structured pruning
from numpy import linalg as LA
import numpy as np

'''
                REDISTRIBUTION
'''

def momentum_redistribution(masking, name, weight, mask):
    """Calculates momentum redistribution statistics.
    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.
        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.
        weight      The weight of the respective sparse layer.
                    This is a torch parameter.
        mask        The binary mask. 1s indicated active weights.
    Returns:
        Layer Statistic      The unnormalized layer statistics
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.
    The calculation of redistribution statistics is the first
    step in this sparse learning library.
    """
    grad = masking.get_momentum_for_weight(weight)
    mean_magnitude = torch.abs(grad)[mask.byte()].mean().item()
    print('mean magnitude: {}'.format(mean_magnitude))
    return mean_magnitude

def magnitude_redistribution(masking, name, weight, mask):
    mean_magnitude = torch.abs(weight)[mask.byte()].mean().item()
    return mean_magnitude

def nonzero_redistribution(masking, name, weight, mask):
    nonzero = (weight !=0.0).sum().item()
    return nonzero

def no_redistribution(masking, name, weight, mask):
    num_params = masking.baseline_nonzero
    n = weight.numel()
    return n/float(num_params)


'''
                PRUNE
'''
def magnitude_prune(masking, mask, weight, name):
    """Prunes the weights with smallest magnitude.
    The pruning functions in this sparse learning library
    work by constructing a binary mask variable "mask"
    which prevents gradient flow to weights and also
    sets the weights to zero where the binary mask is 0.
    Thus 1s in the "mask" variable indicate where the sparse
    network has active weights. In this function name
    and masking can be used to access global statistics
    about the specific layer (name) and the sparse network
    as a whole.
    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.
        mask        The binary mask. 1s indicated active weights.
        weight      The weight of the respective sparse layer.
                    This is a torch parameter.
        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.
    Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed
    Accessable global statistics:
    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]
    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    #souvik: commented the following two lines temporarily
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


    '''
    Souvik: added the following code to prune in a structured way.
    #################
    # Filter pruning
    #################
    '''
def filter_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0
    weight = weight.cpu().detach().numpy()
    shape = weight.shape   
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l2_norm = LA.norm(weight2d, 2, axis=1)
    new_nonzeros = math.ceil(masking.name2nonzeros[name] - num_remove)
    percent = 100*(k/(new_nonzeros+k))
    percentile = np.percentile(row_l2_norm, percent)
    under_threshold = row_l2_norm < percentile
    above_threshold = row_l2_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    expand_above_threshold = expand_above_threshold.reshape(shape)

    return torch.from_numpy(expand_above_threshold)

    '''
    Souvik: added the following code to prune in a structured way.
    #################
    # Channel pruning
    #################
    '''

def channel_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0
    weight = weight.cpu().detach().numpy()
    shape = weight.shape   
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    channel_size = shape[2]*shape[3]
    num_channels = shape[1]
    fro_norm = np.zeros(int(num_channels))
    k_ch = 0
    for i in range (0,int(num_channels)):
        c_start = i*(channel_size)
        c_end = (i+1)*(channel_size)
        chnl_matrix = weight2d[:,c_start:c_end]
        fro_norm[k_ch] = LA.norm(chnl_matrix, 'fro')
        k_ch = k_ch + 1
    new_nonzeros = math.ceil(masking.name2nonzeros[name] - num_remove)
    percent = 100*(k/(new_nonzeros+k))
    percentile = np.percentile(fro_norm, percent)
    under_threshold = fro_norm < percentile
    above_threshold = fro_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(k_ch):
        c_start = i*(channel_size)
        c_end = (i+1)*(channel_size)
        if (above_threshold[i]):
            expand_above_threshold[:,c_start:c_end] = 1
    expand_above_threshold = expand_above_threshold.reshape(shape)

    return torch.from_numpy(expand_above_threshold)

    '''
    Souvik: added the following code to prune in a structured way.
    #################
    # Column pruning
    #################
    '''

def column_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0
    weight = weight.cpu().detach().numpy()
    shape = weight.shape   
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    column_l2_norm = LA.norm(weight2d, 2, axis=0)
    new_nonzeros = math.ceil(masking.name2nonzeros[name] - num_remove)
    percent = 100*(k/(new_nonzeros+k))
    percentile = np.percentile(column_l2_norm, percent)
    under_threshold = column_l2_norm < percentile
    above_threshold = column_l2_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[1]):
        expand_above_threshold[:, i] = above_threshold[i]
    expand_above_threshold = expand_above_threshold.reshape(shape)

    return torch.from_numpy(expand_above_threshold)

'''   
    Incomplete !!! [Pruning is done, growth is not supported yet!]
    Souvik: added the following code to prune in a semi-structured way.
    #################
    # Block pruning
    #################
    
def block_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0
    weight = weight.cpu().detach().numpy()
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    block_size = 16
    num_possible_blocks = int((shape2d[0]*shape2d[1])/(block_size*block_size))
    fro_norm = np.zeros(num_possible_blocks)
    row_size = int(shape2d[0]/block_size)
    col_size = int(shape2d[1]/block_size)
    percent = 100*(k/(new_nonzeros+k))
    k_num = 0
    for i in range(row_size):
        for j in range(col_size):
            r_strt = i*block_size
            r_end = (i + 1)*block_size
            c_strt = j*block_size
            c_end = (j + 1)*block_size
            mini_block = weight2d[r_strt:r_end, c_strt:c_end] 
            fro_norm[k_num] = LA.norm(mini_block, 'fro')
            k_num = k_num + 1
    percentile = np.percentile(fro_norm, percent)
    under_threshold = fro_norm < percentile
    above_threshold = fro_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    j_mask = 0
    for i in range (k_num):
        row2d_strt = j_mask*block_size
        row2d_end = (j_mask+1)*block_size
        col2d_strt = (i%col_size)*block_size
        col2d_end = ((i%col_size) + 1)*block_size
        if((i%col_size)!= 0 or i == 0):
            if(above_threshold[i]):
                expand_above_threshold[row2d_strt:row2d_end, col2d_strt:col2d_end] = 1
        else:
            j_mask = j_mask + 1    
            if(above_threshold[i]):
                expand_above_threshold[row2d_strt:row2d_end, col2d_strt:col2d_end] = 1

    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold)
'''
def global_magnitude_prune(masking):
    prune_rate = 0.0
    for name in masking.name2prune_rate:
        if name in masking.masks:
            prune_rate = masking.name2prune_rate[name]
    tokill = math.ceil(prune_rate*masking.baseline_nonzero)
    total_removed = 0
    prev_removed = 0
    while total_removed < tokill*(1.0-masking.tolerance) or (total_removed > tokill*(1.0+masking.tolerance)):
        total_removed = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.name2nonzeros[name] - remain

        if prev_removed == total_removed: break
        prev_removed = total_removed
        if total_removed > tokill*(1.0+masking.tolerance):
            masking.prune_threshold *= 1.0-masking.increment
            masking.increment *= 0.99
        elif total_removed < tokill*(1.0-masking.tolerance):
            masking.prune_threshold *= 1.0+masking.increment
            masking.increment *= 0.99

    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name][:] = torch.abs(weight.data) > masking.prune_threshold

    return int(total_removed)


def magnitude_and_negativity_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    if num_remove == 0.0: return weight.data != 0.0

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + (num_remove/2.0))

    # remove all weights which absolute value is smaller than threshold
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0

    # remove the most negative weights
    x, idx = torch.sort(weight.data.view(-1))
    mask.data.view(-1)[idx[:math.ceil(num_remove/2.0)]] = 0.0

    return mask

'''
                GROWTH
'''

def random_growth(masking, name, new_mask, total_regrowth, weight):
    n = (new_mask==0).sum().item()
    if n == 0: return new_mask
    expeced_growth_probability = (total_regrowth/n)
    new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
    return new_mask.byte() | new_weights

def momentum_growth(masking, name, new_mask, total_regrowth, weight):
    """Grows weights in places where the momentum is largest.
    Growth function in the sparse learning library work by
    changing 0s to 1s in a binary mask which will enable
    gradient flow. Weights default value are 0 and it can
    be changed in this function. The number of parameters
    to be regrown is determined by the total_regrowth
    parameter. The masking object in conjunction with the name
    of the layer enables the access to further statistics
    and objects that allow more flexibility to implement
    custom growth functions.
    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.
        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.
        new_mask    The binary mask. 1s indicated active weights.
                    This binary mask has already been pruned in the
                    pruning step that preceeds the growth step.
        total_regrowth    This variable determines the number of
                    parameters to regrow in this function.
                    It is automatically determined by the
                    redistribution function and algorithms
                    internal to the sparselearning library.
        weight      The weight of the respective sparse layer.
                    This is a torch parameter.
    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.
    Access to optimizer:
        masking.optimizer
    Access to momentum/Adam update:
        masking.get_momentum_for_weight(weight)
    Accessable global statistics:
    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]
    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    #Souvik: following line added for debug
    print('Total regrowth {}'.format(total_regrowth))
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    
    #Souvik: following two lines are commented temporarily
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
    return new_mask


    #Souvik: following part is added to perform filter growth
    #########################
    # Filter-momentum growth
    #########################
def filter_momentum_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    grad_weight = grad
    grad_weight = grad_weight.cpu().detach().numpy()
    shape = grad_weight.shape   
    grad_weight2d = grad_weight.reshape(shape[0], -1)
    shape2d = grad_weight2d.shape
    row_l2_norm = LA.norm(grad_weight2d, 2, axis=1) 
    percent = 100*(1 - (total_regrowth/(masking.name2nonzeros[name]+masking.name2zeros[name])))
    if (percent < 0):
        percent = 0.0
    if (percent > 100.0):
        percent = 100.0
    percentile = np.percentile(row_l2_norm, percent)
    under_threshold = row_l2_norm < percentile
    above_threshold = row_l2_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    expand_above_threshold = expand_above_threshold.reshape(shape)
    expand_above_threshold = torch.from_numpy(expand_above_threshold).data.byte().cuda()
    new_mask = new_mask +  expand_above_threshold
    return new_mask

    #Souvik: following part is added to perform Channel growth
    #########################
    # Channel-momentum growth
    #########################
def channel_momentum_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    grad_weight = grad
    grad_weight = grad_weight.cpu().detach().numpy()
    shape = grad_weight.shape   
    # import pdb; pdb.set_trace()
    grad_weight2d = grad_weight.reshape(shape[0], -1)
    shape2d = grad_weight2d.shape
    channel_size = shape[2]*shape[3]
    num_channels = shape[1]
    fro_norm = np.zeros(int(num_channels))
    k = 0
    for i in range (0,int(num_channels)):
        c_start = i*(channel_size)
        c_end = (i+1)*(channel_size)
        chnl_matrix = grad_weight2d[:,c_start:c_end]
        fro_norm[k] = LA.norm(chnl_matrix, 'fro')
        k = k+1
    percent = 100*(1 - (total_regrowth/(masking.name2nonzeros[name]+masking.name2zeros[name])))
    if (percent < 0):
        percent = 0.0
    if (percent > 100.0):
        percent = 100.0
    percentile = np.percentile(fro_norm, percent)
    under_threshold = fro_norm < percentile
    above_threshold = fro_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(k):
        c_start = i*(channel_size)
        c_end = (i+1)*(channel_size)
        if (above_threshold[i]):
            expand_above_threshold[:,c_start:c_end] = 1
    expand_above_threshold = expand_above_threshold.reshape(shape)
    expand_above_threshold = torch.from_numpy(expand_above_threshold).data.byte().cuda()
    new_mask = new_mask +  expand_above_threshold
    return new_mask


    #Souvik: following part is added to perform Column growth
    #########################
    # Column-momentum growth
    #########################
def column_momentum_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    grad_weight = grad
    grad_weight = grad_weight.cpu().detach().numpy()
    shape = grad_weight.shape   
    grad_weight2d = grad_weight.reshape(shape[0], -1)
    shape2d = grad_weight2d.shape
    column_l2_norm = LA.norm(grad_weight2d, 2, axis=0) 
    percent = 100*(1 - (total_regrowth/(masking.name2nonzeros[name]+masking.name2zeros[name])))
    if (percent < 0):
        percent = 0.0
    if (percent > 100.0):
        percent = 100.0
    percentile = np.percentile(column_l2_norm, percent)
    under_threshold = column_l2_norm < percentile
    above_threshold = column_l2_norm > percentile
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[1]):
        expand_above_threshold[:, i] = above_threshold[i]
    expand_above_threshold = expand_above_threshold.reshape(shape)
    expand_above_threshold = torch.from_numpy(expand_above_threshold).data.byte().cuda()
    new_mask = new_mask +  expand_above_threshold
    return new_mask

'''
Incomplete!!!
#Souvik: following part is added to perform Block growth
    #########################
    # Block-momentum growth
    #########################
def block_momentum_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    grad_weight = grad
    grad_weight = grad_weight.cpu().detach().numpy()
    shape = grad_weight.shape   
    grad_weight2d = grad_weight.reshape(shape[0], -1)
    shape2d = grad_weight2d.shape
    percent = 100*(1 - (total_regrowth/(masking.name2nonzeros[name]+masking.name2zeros[name])))
    if (percent < 0):
        percent = 0.0
    if (percent > 100.0):
        percent = 100.0
'''

def momentum_neuron_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)

    M = torch.abs(grad)
    if len(M.shape) == 2: sum_dim = [1]
    elif len(M.shape) == 4: sum_dim = [1, 2, 3]

    v = M.mean(sum_dim).data
    v /= v.sum()

    slots_per_neuron = (new_mask==0).sum(sum_dim)

    M = M*(new_mask==0).float()
    for i, fraction  in enumerate(v):
        neuron_regrowth = math.floor(fraction.item()*total_regrowth)
        available = slots_per_neuron[i].item()

        y, idx = torch.sort(M[i].flatten())
        if neuron_regrowth > available:
            neuron_regrowth = available
        # TODO: Work into more stable growth method
        threshold = y[-(neuron_regrowth)].item()
        if threshold == 0.0: continue
        if neuron_regrowth < 10: continue
        new_mask[i] = new_mask[i] | (M[i] > threshold)

    return new_mask


def global_momentum_growth(masking, total_regrowth):
    togrow = total_regrowth
    total_grown = 0
    last_grown = 0
    while total_grown < togrow*(1.0-masking.tolerance) or (total_grown > togrow*(1.0+masking.tolerance)):
        total_grown = 0
        total_possible = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue

                new_mask = masking.masks[name]
                grad = masking.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                possible = (grad !=0.0).sum().item()
                total_possible += possible
                grown = (torch.abs(grad.data) > masking.growth_threshold).sum().item()
                total_grown += grown
        if total_grown == last_grown: break
        last_grown = total_grown


        if total_grown > togrow*(1.0+masking.tolerance):
            masking.growth_threshold *= 1.02
            #masking.growth_increment *= 0.95
        elif total_grown < togrow*(1.0-masking.tolerance):
            masking.growth_threshold *= 0.98
            #masking.growth_increment *= 0.95

    total_new_nonzeros = 0
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue

            new_mask = masking.masks[name]
            grad = masking.get_momentum_for_weight(weight)
            grad = grad*(new_mask==0).float()
            masking.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > masking.growth_threshold)).float()
            total_new_nonzeros += new_mask.sum().item()
    return total_new_nonzeros

#Souvik: prune function decides the prune mode of the layers, this pruning includes whole pruning, not 
# incremental.
prune_funcs = {}
prune_funcs['magnitude'] = magnitude_prune
prune_funcs['SET'] = magnitude_and_negativity_prune
prune_funcs['global_magnitude'] = global_magnitude_prune
#Souvik: added following pruning modes for structured/semi-structured pruning.
prune_funcs['filter'] = filter_prune
prune_funcs['column'] = column_prune
prune_funcs['channel'] = channel_prune


#Souvik: Growth function decides the re-growth mode of the fraction of the 0 weights to become alive, at the end of each epoch
# The growth function is incremental, unlike the pruning function
growth_funcs = {}
growth_funcs['random'] = random_growth
growth_funcs['momentum'] = momentum_growth
growth_funcs['momentum_neuron'] = momentum_neuron_growth
growth_funcs['global_momentum_growth'] = global_momentum_growth
#Souvik: added following growth modes for structured/semi-structured pruning.
growth_funcs['filter_momentum'] = filter_momentum_growth
growth_funcs['column_momentum'] = column_momentum_growth
growth_funcs['channel_momentum'] = channel_momentum_growth



#Souvik: redistribution func is to find the ratio that which layer should get how much share of the regrowth parameters.
redistribution_funcs = {}
redistribution_funcs['momentum'] = momentum_redistribution
redistribution_funcs['nonzero'] = nonzero_redistribution
redistribution_funcs['magnitude'] = magnitude_redistribution
redistribution_funcs['none'] = no_redistribution
