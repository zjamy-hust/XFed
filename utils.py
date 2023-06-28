#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import os
import shutil
import time
from torchvision import datasets, transforms
from datasets.sampling_partition import cifar_iid, cifar_noniid, cifar100_iid, cifar100_noniid, mnist_iid, mnist_noniid
import numpy as np

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = './data/'
    # shared_data = 0
    # if args.shared_data>0:
    #     shared_data = args.shared_data
    # print('zjamy shared data',shared_data)

    cifar_train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    mnist_train_transform = transforms.Compose(             #是否需要进行数据增强？？？？？？？？？？？？？
            [transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.RandomCrop(28, padding=4), 
             transforms.Normalize([0.1307], [0.3081])])
    mnist_apply_transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize([0.1307], [0.3081])])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=cifar_train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=cifar_apply_transform)
        if args.iid:
        # Sample IID user data 
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
        # Sample Non-IID user data
            user_groups = cifar_noniid(train_dataset, args.num_users, args.partition)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=cifar_train_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=cifar_apply_transform)
        if args.iid:
        # Sample IID user data 
            user_groups = cifar100_iid(train_dataset, args.num_users)
        else:
        # Sample Non-IID user data
            user_groups = cifar100_noniid(train_dataset, args.num_users, args.partition)
    elif args.dataset == 'MNIST':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=mnist_train_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=mnist_apply_transform)
        if args.iid:
        # Sample IID user data 
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
        # Sample Non-IID user data
            user_groups = mnist_noniid(train_dataset, args.num_users, args.partition)
    # check proportion of shared_data

    # sample training data amongst users
    
    return train_dataset, test_dataset, user_groups


def average_weights(w,selected_list,selected_client_dataset_sizes=None):
    """
    Returns the average of the weights.
    
    client_dataset_sizes: list。被选中客户端数据集的样本数量，用于设置模型聚合权重。
    """
    if selected_client_dataset_sizes is None: #没有指定样本数量，则平均每个客户端的参数
        w_avg = copy.deepcopy(w[selected_list[0]])
        for key in w_avg.keys():
            # for i in range(1, len(w)):
            for i in selected_list[1:]:
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(selected_list))
    else:
        if not isinstance(selected_client_dataset_sizes,list):
            raise ValueError("selected_client_dataset_sizes必须为list。")
        if not len(selected_client_dataset_sizes) == len(selected_list):
            raise ValueError("selected_client_dataset_sizes的长度需要与selected_list一致。")
        selected_client_weights = np.array(selected_client_dataset_sizes)/sum(selected_client_dataset_sizes)
        w_avg = copy.deepcopy(w[selected_list[0]])
        for key in w_avg.keys():
            # for i in range(1, len(w)):
            if "weight" in key or "bias" in key:
                w_avg[key] *= selected_client_weights[selected_list[0]]
                for i in selected_list[1:]:
                    w_avg[key] += w[i][key] * selected_client_weights[i]
                w_avg[key] = torch.div(w_avg[key], len(selected_list))
            else:
                for i in selected_list[1:]:
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(selected_list))
    return w_avg

def average_weights_for_model_with_global_mask(w,selected_list,selected_client_dataset_sizes=None):
    """
    Returns the average of the weights.
    
    client_dataset_sizes: list。被选中客户端数据集的样本数量，用于设置模型聚合权重。
    """
    if selected_client_dataset_sizes is None: #没有指定样本数量，则平均每个客户端的参数
        w_avg = copy.deepcopy(w[selected_list[0]])
        for key in w_avg.keys():
            if "backbone_1" in key:
                continue
            
            for i in selected_list[1:]:
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(selected_list))
            
            if "backbone_2" in key:
                key_splited = key.split(".")
                key_splited[0] = "backbone_1"
                new_key = ".".join(key_splited)
                w_avg[new_key] = copy.deepcopy(w_avg[key])
    else:
        if not isinstance(selected_client_dataset_sizes,list):
            raise ValueError("selected_client_dataset_sizes必须为list。")
        if not len(selected_client_dataset_sizes) == len(selected_list):
            raise ValueError("selected_client_dataset_sizes的长度需要与selected_list一致。")
        selected_client_weights = np.array(selected_client_dataset_sizes)/sum(selected_client_dataset_sizes)
        w_avg = copy.deepcopy(w[selected_list[0]])
        for key in w_avg.keys():
            if "backbone_1" in key:
                continue
            
            if "weight" in key or "bias" in key:
                w_avg[key] *= selected_client_weights[selected_list[0]]
                for i in selected_list[1:]:
                    w_avg[key] += w[i][key] * selected_client_weights[i]
                w_avg[key] = torch.div(w_avg[key], len(selected_list))
            else:
                for i in selected_list[1:]:
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(selected_list))
                
            if "backbone_2" in key:
                key_splited = key.split(".")
                key_splited[0] = "backbone_1"
                new_key = ".".join(key_splited)
                w_avg[new_key] = copy.deepcopy(w_avg[key])
            
    return w_avg


def weights_norm_2L_regularization(global_model_dict, local_model_dict):
    local_model_ = copy.deepcopy(local_model_dict)
    global_model_ = copy.deepcopy(global_model_dict)
    
    regularization_num=0
    for key in local_model_.keys():
        if "weight" in key or "bias" in key:
            regularization_num += torch.norm(global_model_[key] - local_model_[key], p=2)
    return regularization_num


def exp_details(log,args):
    log.logger.debug(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    log.logger.debug('\nExperimental details:')
    log.logger.debug(f'    Model     : {args.model}')
    log.logger.debug(f'    Optimizer : {args.optimizer}')
    log.logger.debug(f'    Learning  : {args.lr}')
    log.logger.debug(f'    Global Rounds   : {args.epochs}\n')

    log.logger.debug('    Federated parameters:')
    if args.iid:
        log.logger.debug('    IID')
    else:
        log.logger.debug('    Non-IID')
    #print(f'    Fraction of users  : {args.frac}')
    log.logger.debug(f'    Local Batch size   : {args.local_bs}')
    log.logger.debug(f'    Local Epochs       : {args.local_ep}\n')
    
    if args.num_users:
        log.logger.debug(f'    num_users   : {args.num_users}')
    # if args.shared_data:
    #     log.logger.debug(f'    shared_data   : {args.shared_data}')
    if args.pretrained_model :
        log.logger.debug(f'    pretrained_model   : {args.pretrained_model}')
    return

def save_checkpoint(args, state, is_best, local_idx, is_global):
    if is_global == 0:
        if not os.path.isdir(f'save_checkpoints/{args.model}_local/'):
            os.makedirs(f'save_checkpoints/{args.model}_local/')
        filename = f'save_checkpoints/{args.model}_local/local_{local_idx}.gpu{args.gpu}.mode{args.mode}.global_epoch{args.epochs}.local_ep{args.local_ep}.num_users{args.num_users}.ckpt.pth.tar'
    else:
        if not os.path.isdir(f'save_checkpoints/{args.model}_global/'):
            os.makedirs(f'save_checkpoints/{args.model}_global/')
        filename = f'save_checkpoints/{args.model}_global/global.iid{args.iid}.gpu{args.gpu}.mode{args.mode}.global_epoch{args.epochs}.local_ep{args.local_ep}.num_users{args.num_users}.ckpt.pth.tar'
    torch.save(state, filename)
    print(f'saved checkpoint to {filename}')
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
        print(f'saved checkpoint to {filename}')
