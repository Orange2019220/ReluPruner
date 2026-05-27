from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_augmix
from dataset.cifar10_deepaug import TriComprehensiveRobustnessDataloader
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_augmix
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample
from dataset.tiny_imagenet import get_tiny_imagenet_dataloaders, get_tiny_imagenet_dataloaders_augmix
from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, train_adversarial, train_comp_robust, validate, validate_dualBN, train_adversarial_dualmask
from helper.loops_robust import train_adversarial_3BN, train_comp_robust_3BN, validate_3BN
from helper.attack import PGD, FGSM
from helper.prune_utils import Masking

from distiller_zoo import DistillKL

import sys

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')


    # models (note there are two resnet50s, one from actual CRD paper,
    # the other written by us to match the param count of Stand-alone SA (SASA) paper
    # The resnet50, resnet38, resnet26 are matching the model of SASA paper
    parser.add_argument('--model', type=str, default='resnet56',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'ResNet18', 'ResNet34', 'ResNet50', 'wrn_16_1', 'wrn_16_2', 'ResNet18_3BN',
                                 'wrn_40_1', 'wrn_40_2', 'resnet50', 'resnet38', 'resnet26',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18Prelu',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomResNet18', 'wrn_22_8', 'wrn_22_8_3BN'])
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tiny_imagenet', 'imagenet'], help='dataset')

    # attack parameters
    parser.add_argument('--attack_mode', type=str, default=None, choices=['pgd', 'fgsm'])
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--steps', type=int, default=7)
    parser.add_argument('--_lambda', type=int, default=None, choices=[1, 6]) # hyperparameter for TRADES Loss
    parser.add_argument('--robust_train_mode', type=str, default='pgd', choices=['trades', 'pgd'])
    parser.add_argument('--augment_mode', type=str, default=None, choices=['augmix', 'deepaug'])
    parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
    parser.add_argument('--use3BN', action='store_true', help='If true, use triple BN')
    parser.add_argument('--weight_masking', action='store_true', help='If true, separate weights for nat and adv')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument("--seed", help="set random generator seed", default='0')
    parser.add_argument("--local-rank", default=0, type=int)

    opt = parser.parse_args()
    torch.manual_seed(int(opt.seed))
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay)
    if opt.attack_mode: opt.model_name += opt.attack_mode

    # save_folder_name = '{}_{}_lr_{}_{}_lambda_{}'.format(opt.model, opt.dataset, opt.learning_rate,
    #                                                     opt.robust_train_mode, opt._lambda)
    # save_folder_name = '{}_{}_batch_{}_lr_{}_gpu:2'.format(opt.model, opt.dataset, opt.batch_size, opt.learning_rate)
    save_folder_name = '{}_{}'.format(opt.model, opt.dataset)
    if opt.augment_mode:
        save_folder_name += '_' + opt.augment_mode
    # if opt.attack_mode:
    #     save_folder_name += '_' + opt.attack_mode
    if opt.attack_mode:
        save_folder_name += '_' + opt.attack_mode
    if opt.use2BN:
        save_folder_name += '_2BN'
    if opt.use3BN:
        save_folder_name += '_3BN' 
    if opt.weight_masking:
        save_folder_name += '_wt_mask'
    # save_folder_name += '_'+str(opt.seed)
    # save_folder_name += '_loss3'
    save_folder_name += '_batch{}'.format(opt.batch_size)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.tb_folder = os.path.join(opt.tb_path, save_folder_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, save_folder_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # save terminal output to a file
    opt.outfile = os.path.join(opt.save_folder,'results.out')

    return opt


def main():
    best_acc = 0
    best_adv_acc = 0

    opt = parse_option()
    # f = open(opt.outfile, 'w')
    # sys.stdout = f
    print(opt)

    # distributed settings
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ:
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
    opt.device = 'cuda:0'
    opt.world_size = 1
    opt.rank = 0  # global rank
    # print (f"int(os.environ['WORLD_SIZE']):{int(os.environ['WORLD_SIZE'])}")
    print ('Distributed training: ',opt.distributed)
    if opt.distributed:
        opt.device = 'cuda:%d' % opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = torch.distributed.get_rank()
        #logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
        #             % (opt.rank, opt.world_size))
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d %d, total %d.'
                     % (opt.rank, opt.local_rank, opt.world_size))
    else:
        #logger.info('Training with a single process on 1 GPUs.')
        print('Training with a single process on 1 GPUs.')
    assert opt.rank >= 0

    criterion_div = DistillKL(opt.kd_T)
    attacker = None
    attacker_val = None
    if opt.attack_mode == 'pgd':
        if opt.robust_train_mode == 'trades':
            attacker = PGD(eps=opt.eps, steps=opt.steps, criterion=criterion_div)
        else:
            attacker = PGD(eps=opt.eps, steps=opt.steps)
        attacker_val = PGD(eps=opt.eps, steps=opt.steps)
    elif opt.attack_mode == 'fgsm': attacker = FGSM(eps=opt.eps)
        
    # dataloader
    if opt.dataset == 'cifar100':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader = get_cifar100_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader = get_cifar10_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        elif opt.augment_mode == 'deepaug':
            train_loader, val_loader = TriComprehensiveRobustnessDataloader(dataset='cifar10', batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        else:    
            train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        n_cls = 10
    elif opt.dataset == 'tiny_imagenet':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader = get_tiny_imagenet_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        else:
            train_loader, val_loader = get_tiny_imagenet_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed)
        n_cls=200
    elif opt.dataset == 'imagenet':
        if opt.distributed:
            train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        distributed=opt.distributed)
        else:
            train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
        n_cls=1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.use2BN:
        model = model_dict[opt.model](num_classes=n_cls, use2BN=opt.use2BN)
    elif opt.use3BN:
        model = model_dict[opt.model](num_classes=n_cls, use3BN=opt.use3BN)
    else:
        model = model_dict[opt.model](num_classes=n_cls)


    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank])

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # weight separation
    mask_weight = None
    # if opt.weight_masking:
    #     mask_weight = Masking(optimizer=optimizer)
    #     mask_weight.add_module(model, density=0.5)

    if opt.use2BN:
        val_lambdas = [0.0, 1.0]
        best_TA, best_ATA = {}, {}
        for val_lambda in val_lambdas:
            best_TA[val_lambda], best_ATA[val_lambda] = 0, 0
    if opt.use3BN:
        val_lambdas = [0.0, 2.0]
        best_TA, best_ATA = {}, {}
        for val_lambda in val_lambdas:
            best_TA[val_lambda], best_ATA[val_lambda] = 0, 0
    # routine
    for epoch in range(1, opt.epochs + 1):
        
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if attacker:
            if opt.augment_mode == 'deepaug':
                if opt.use3BN:
                    train_acc, robust_acc, train_loss = train_comp_robust_3BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
                else:
                    train_acc, robust_acc, train_loss = train_comp_robust(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
            else:
                if opt.use3BN:
                    train_acc, robust_acc, train_loss = train_adversarial_3BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
                elif opt.use2BN and opt.weight_masking:
                    train_acc, robust_acc, train_loss = train_adversarial_dualmask(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
                    # train_acc, robust_acc, train_loss = train_adversarial_2BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
                else:
                    train_acc, robust_acc, train_loss = train_adversarial(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
        else:
            train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        # train_acc, robust_acc, train_loss = train_adversarial_2BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if opt.local_rank == 0:
            if attacker:
                logger.log_value('robust_acc', robust_acc, epoch)
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        time_val_start = time.time()
        if opt.use2BN:
            test_acc, robust_test_acc, _ = validate_dualBN(val_loader, model, criterion, opt, attacker=attacker_val)
        elif opt.use3BN:
            test_acc, robust_test_acc, _ = validate_3BN(val_loader, model, criterion, opt, attacker=attacker_val)
        else:
            test_acc, robust_test_acc, test_loss, _ = validate(val_loader, model, criterion, opt, attacker=attacker_val)

        # test_acc, robust_test_acc, test_loss, _ = validate_2BN(val_loader, model, criterion, opt, attacker=attacker_val)

        if opt.local_rank == 0:
            if opt.use2BN or opt.use3BN:
                val_str = 'Epoch %d | Validation | Time: %.4f\n' % (epoch, (time.time()-time_val_start))
                for val_lambda in val_lambdas:
                    val_str += 'val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, test_acc[val_lambda].avg, robust_test_acc[val_lambda].avg)
                    if test_acc[val_lambda].avg >= best_TA[val_lambda]:
                        best_TA[val_lambda] = test_acc[val_lambda].avg # update best TA
                        # if opt.weight_masking:
                        #     state = {
                        #     'epoch': epoch,
                        #     'model': model.state_dict(),
                        #     'best_acc': best_TA[val_lambda],
                        #     'robust_acc': robust_test_acc[val_lambda].avg, 
                        #     'mask_c': mask_weight.masks_c,
                        #     'mask_a': mask_weight.masks_a
                        # }
                        # else:    
                        state = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'best_acc': best_TA[val_lambda],
                            'robust_acc': robust_test_acc[val_lambda].avg, 
                        }

                        # state = {
                        #     'epoch': epoch + 1,
                        #     'state_dict': self.state_dict(),
                        #     'best_prec1': best_prec1,
                        #     'is_best': is_best
                        # }

                        save_file = os.path.join(opt.save_folder, '{}_best_TA{}.pth'.format(opt.model, val_lambda))
                        print('saving the best model!')
                        torch.save(state, save_file)
                    if robust_test_acc[val_lambda].avg >= best_ATA[val_lambda]:
                        best_ATA[val_lambda] = robust_test_acc[val_lambda].avg # update best ATA
                        # if opt.weight_masking:
                        #     state = {
                        #     'epoch': epoch,
                        #     'model': model.state_dict(),
                        #     'best_robust_acc': best_ATA[val_lambda],
                        #     'natural acc': test_acc[val_lambda].avg,
                        #     'mask_c': mask_weight.masks_c,
                        #     'mask_a': mask_weight.masks_a
                        # }
                        # else:
                        state = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'best_robust_acc': best_ATA[val_lambda],
                            'natural acc': test_acc[val_lambda].avg,
                        }
                        save_file = os.path.join(opt.save_folder, '{}_best_ATA{}.pth'.format(opt.model, val_lambda))
                        print('saving the best robust model!')
                        torch.save(state, save_file)
                
                val_fp = open(os.path.join(opt.save_folder, 'val_log.txt'), 'a+')
                print(val_str)
                val_fp.write(val_str + '\n')
                val_fp.close()
            else:
                if attacker:
                    logger.log_value('robust_test_acc', robust_test_acc, epoch)
                logger.log_value('test_acc', test_acc, epoch)
                # logger.log_value('test_acc_top5', test_acc_top5, epoch)
                logger.log_value('test_loss', test_loss, epoch)

                # save the best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    state = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'best_acc': best_acc,
                        'robust_acc': robust_test_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
                    print('saving the best model!')
                    torch.save(state, save_file)

                if attacker:
                    if robust_test_acc > best_adv_acc:
                        best_adv_acc = robust_test_acc
                        state = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'best_robust_acc': best_adv_acc,
                            'natural_acc': test_acc,
                            'optimizer': optimizer.state_dict(),
                        }
                        save_file = os.path.join(opt.save_folder, 'robust_{}_best.pth'.format(opt.model))
                        print('saving the best robust model!')
                        torch.save(state, save_file)

            # regular saving
            # if epoch % opt.save_freq == 0:
            #     print('==> Saving...')
            #     state = {
            #         'epoch': epoch,
            #         'model': model.state_dict(),
            #         'accuracy': test_acc,
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    if opt.local_rank == 0:
        if opt.use2BN or opt.use3BN:
            for val_lambda in val_lambdas:
                print('Best Accuracies:')
                print('val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, best_TA[val_lambda], best_ATA[val_lambda]))
        else:
            print('best accuracy:', best_acc)
            if attacker:
                print('best robust accuracy:', best_adv_acc)

        # save model
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
        torch.save(state, save_file)

    # f.close()
    
if __name__ == '__main__':
    main()
