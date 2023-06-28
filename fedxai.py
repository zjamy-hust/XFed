#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gc
import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from utils import get_dataset, average_weights, exp_details, save_checkpoint, average_weights_for_model_with_global_mask
from localupdates.update import LocalUpdate, test_inference, test_inference_with_mask, generate_dataset_mask, test_inference_with_global_mask
from models.models import TestmyNet
from models.models_resnet import ResNet18, ResNet18_with_mask
from models.models_shufflenetv2 import ShuffleNetV2
from models.models_resnext import resnext
from operator import itemgetter, attrgetter
# from xai import XAI_evaluate, XAI_evaluate_with_global_masks
from xai import XAI_evaluate_with_global_masks
from pathlib import Path
best_local_acc = 0
best_global_acc = 0
pretrained_model='./checkpoints/pre_ckpt.best.pth.tar'

import logging
from logging import handlers
 
print("dance:", os.getpid())

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
 
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
 
    def __init__(self,filename,level='info',when='D',interval=30,backCount=100,fmt='%(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        self.logger.propagate = False

        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,interval=interval,backupCount=backCount,encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(th)

if __name__ == '__main__':
    log = Logger('mylog/'+time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())+'.log',level='debug')

    best_local_acc = 0
    best_global_acc = 0
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs/'+time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    args = args_parser()
    
    set_random_seed(args.random_seed)       
    
    exp_details(log,args)
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    if args.iid==0:
        log.logger.debug(f'\n')
        for j in range(args.num_users):
            # print(len(user_groups[j]))
            log.logger.debug(f"    user_groups['{j}'] '{len(user_groups[j])}'")

    asset_path='assets'
    if args.dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
        #            'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7, 'ship' 8, 'truck' 9)
        XAI_labels=[7, 8, 2, 2, 0, 5, 7, 9, 2, 8, 8, 2, 8, 2, 5, 8, 0, 7, 5, 5,1,1,3,3,4,4,6,6,9,3 ]
        assetpath = str(Path(asset_path)/'cifar_asset')
        print("assetpath",assetpath)
        files = os.listdir(assetpath)        
    elif args.dataset == "MNIST":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        assetpath = str(Path(asset_path)/'mnist_asset')
        with open(assetpath+"/labels.txt",'r') as f:
            read_res = f.readlines()
            XAI_labels = [int(line.strip()) for line in read_res]
        print("XAI labels", XAI_labels)
        print("assetpath",assetpath)
        files = os.listdir(assetpath)        
    
    
    # BUILD MODEL
    if args.dataset == "cifar10":
        num_classes = 10
        input_channel = 3
    elif args.dataset == "MNIST":
        num_classes = 10
        input_channel = 1
        
    if args.model == 'test':
        global_model = TestmyNet()
    elif args.model == 'resnet18':
        if args.mode in [6,7,8]:
            global_model = ResNet18_with_mask(num_classes, input_channel,True if args.output_hard_mask==1 else False)
        else:
            global_model = ResNet18(num_classes, input_channel)
    elif args.model == 'shufflenetv2':
        if args.dataset == "cifar10":
            global_model = ShuffleNetV2(1)      
        else:
            raise ValueError("args.dataset error。")
    elif args.model == 'resnext':
        if args.dataset == "cifar10":           
            global_model = resnext(cardinality=8,num_classes=100,depth=29,widen_factor=4,dropRate=0)
        else:
            raise ValueError("args.dataset error。")
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    log.logger.debug(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    global_model.train()
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    start_epoch = 0
    val_loss_pre, counter = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            log.logger.debug(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            start_epoch = checkpoint['epoch']
            global_model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            train_accuracy = checkpoint['train_accuracy']
            print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            log.logger.debug(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"No checkpoint found at '{args.resume}'")    
    else :
        if args.pretrained_model:
            checkpoint = torch.load(args.pretrained_model, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            #checked with res18 to res18, res18 to vgg,  looks like issues with res18 to shufflenetv2
            if 'moco_ckpt' not in args.pretrained_model:
                #this is for rot pretrain
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if 'linear' not in k and 'fc' not in k:
                        new_state_dict[k] = v
                global_model.load_state_dict(new_state_dict, strict=False)
                print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
                log.logger.debug(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
            print(f'===> Pre-trained model loaded: {args.pretrained_model}')
            log.logger.debug(f'===> Pre-trained model loaded: {args.pretrained_model}')

    best_test_acc = 0
    is_best = 0
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), args.lr,
                                    momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), args.lr,
                                     weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    test_loss_list = []
    test_acc_list = []
    test_in_mask_acc_mean_list = []
    test_XAI_ACC_list = []

    for epoch in range(start_epoch, start_epoch+args.epochs):
        epoch_start_time = time.time()
        #D
        is_best = 0
        local_weights, local_losses= [], []
        #update learning rate
        args.lr=scheduler.get_last_lr()[0]
        print(f'\n | Global Training Round : {epoch+1} |\n')
        log.logger.debug(f'\n | Global Training Round : {epoch+1} |\n')
        
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        start_user = 0
        # Set optimizer for the local updates
        local_test_acc_list = []
        local_test_loss_list = []
        local_XAI_acc_list = []
        in_mask_acc_mean_list = []
        client_dataset_size = {}    
        for idx in range(start_user,args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            client_dataset_size[idx] = len(user_groups[idx])
            # model=copy.deepcopy(global_model)
            
            local_init_model=copy.deepcopy(global_model)
            
            if args.mode in [0,2,5]:  
                w, loss, lw  = local_model.update_weights(
                    model=local_init_model, global_round=epoch, user=idx)
            elif args.mode == 1:     
                if epoch < args.mode1_start_epoch:       
                    w, loss, lw  = local_model.update_weights(
                        model=local_init_model, global_round=epoch, user=idx)     
                else:
                
                    print("Use Data Augmentation.")
                    train_masks = generate_dataset_mask(local_init_model,
                                                        dataset=train_dataset,
                                                        idxs=user_groups[idx],
                                                        batch_size=args.train_mask_batch_size,
                                                        nt_samples=args.train_mask_nt_samples,
                                                        n_steps=args.train_mask_n_steps,
                                                        device=device,
                                                        topk = args.topk)          
                    w, loss, lw  = local_model.update_weights_augmentation(
                        model=local_init_model, global_round=epoch, user=idx, train_masks = train_masks) 
            elif args.mode == 4:        #FedProx
                w, loss, lw  = local_model.update_weights_fedprox(
                    model=local_init_model, global_round=epoch, user=idx, global_model=global_model)
            elif args.mode == 3:
                w, loss, lw  = local_model.update_weights_augmentation_similarity(model=local_init_model,
                                                                                  device=device,
                                                                                  global_round=epoch, 
                                                                                  user=idx, 
                                                                                  train_mask_batch_size=args.mode3_train_mask_batch_size,
                                                                                  train_mask_nt_samples=args.mode3_train_mask_nt_samples,
                                                                                  train_mask_n_steps=args.mode3_train_mask_n_steps,
                                                                                  topk=args.topk,
                                                                                  mse_loss_lambda=args.mse_loss_lambda,
                                                                                  mapping = args.mapping)
            elif args.mode == 6:
                w, loss, lw  = local_model.update_weights_with_global_mask(     #
                    model=local_init_model, global_round=epoch, user=idx)
            else:
                raise ValueError("args.mode error。")
            
            #simulate this will happen in the enclave or cloud side
            if args.mode in [0,1,3,4,5]:  
                local_test_acc, local_test_loss =  test_inference(args, model=copy.deepcopy(lw), test_dataset=test_dataset)            
            elif args.mode == 2:
                if epoch < args.mode2_end_epoch:
                    test_masks = generate_dataset_mask(local_init_model,
                                                        dataset=test_dataset,
                                                        idxs=[i for i in range(len(test_dataset))],
                                                        batch_size=args.test_mask_batch_size,
                                                        nt_samples=args.test_mask_nt_samples,
                                                        n_steps=args.test_mask_n_steps,
                                                        device=device,
                                                        topk = args.topk)     
                local_test_acc, local_test_loss =  test_inference_with_mask(args, model=copy.deepcopy(lw), test_dataset=test_dataset, test_masks=test_masks)
            elif args.mode == 6:
                local_test_acc, local_test_loss =  test_inference_with_global_mask(
                    args, model=copy.deepcopy(lw), test_dataset=test_dataset) 
            
            logger.add_scalar(f"user{idx}_test_acc", local_test_acc, epoch)
            local_test_acc_list.append((idx,local_test_acc))
            local_test_loss_list.append((idx,local_test_loss))

           
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if args.save_local == 1:
                    save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'arch': args.model,
                        'state_dict': w,
                    }, is_best, idx, is_global=0)
            print(f'Global:{epoch}, user:{idx}, size:{len(user_groups[idx])} loss: {loss:.4f}')
            log.logger.debug(f'Global:{epoch}, user:{idx}, size:{len(user_groups[idx])} loss: {loss:.4f}')
            optimizer.step() 

        #####client selection to be added here######            
        
        #select top 12 acc & loss
        #select top 12 XAI data
        #selected = 1           
        # Initializing N 
        N = args.client_selection_num
        # Get Top N elements from Records
        # Using sorted() + itemgetter()
        
        
        if args.mode in [0,6]:      
            res = random.sample(local_test_acc_list,N)
        elif args.mode in [1,2,3]:
            # Here is Ablation Study
            # mean_acc_and_XAI_acc = [(local_test_acc_list[i][0],(0.8*local_test_acc_list[i][1]+0.2*local_XAI_acc_list[i][1])) for i in range(len(local_test_acc_list))] 
            mean_acc_and_XAI_acc = [(local_test_acc_list[i][0],(local_test_acc_list[i][1]+local_XAI_acc_list[i][1])/2) for i in range(len(local_test_acc_list))]   
            res = sorted(mean_acc_and_XAI_acc, key=itemgetter(1), reverse = True)[:N]
        elif args.mode in [4,5]:
            res = sorted(local_test_acc_list, key=itemgetter(1), reverse = True)[:N]
        print("The sorted acc_list is : " + str(res))
        
        selected_client_idx_list = []
        for item in res:
            selected_client_idx_list.append(item[0])
        print("The selected client list is : " + str(selected_client_idx_list))
        #####client selection end####################

        # update global weights
        if args.is_aggregate_with_weights == 0: 
            if args.mode in [6]:
                global_weights = average_weights_for_model_with_global_mask(local_weights,selected_client_idx_list)
            else:
                global_weights = average_weights(local_weights,selected_client_idx_list)
        elif args.is_aggregate_with_weights == 1: 
            if args.mode in [6]:
                global_weights = average_weights_for_model_with_global_mask(local_weights,
                                                                            selected_client_idx_list,
                                                                            [client_dataset_size[i] for i in range(len(local_weights)) if i in selected_client_idx_list])
            else:
                global_weights = average_weights(local_weights,
                                                selected_client_idx_list,
                                                [client_dataset_size[i] for i in range(len(local_weights)) if i in selected_client_idx_list])
        else:
            raise ValueError("args.mode有误。")

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[c], logger=logger)
        #     acc, loss, _ = local_model.inference(model=global_model,global_round=1000,user=c)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if args.mode in [0,1,3,4,5]: 
            test_acc, test_loss =  test_inference(args, global_model, test_dataset)   
        elif args.mode == 2:
            if epoch < args.mode2_end_epoch:
                test_masks = generate_dataset_mask(global_model,
                                                    dataset=test_dataset,
                                                    idxs=[i for i in range(len(test_dataset))],
                                                    batch_size=args.test_mask_batch_size,
                                                    nt_samples=args.test_mask_nt_samples,
                                                    n_steps=args.test_mask_n_steps,
                                                    device=device,
                                                    topk = args.topk)     
            test_acc, test_loss =  test_inference_with_mask(args, model=global_model, test_dataset=test_dataset, test_masks=test_masks)
        elif args.mode == 6:
            test_acc, test_loss =  test_inference_with_global_mask(
                args, global_model, test_dataset=test_dataset) 
        
        if args.mode in [0,1,3,4,5,6]:  
            in_mask_acc_mean,out_mask_acc_mean,XAI_ACC=XAI_evaluate_with_global_masks(global_model,
                                                                                            files,
                                                                                            assetpath,
                                                                                            dataset_name=args.dataset,
                                                                                            device=device,
                                                                                            XAI_labels=XAI_labels,
                                                                                            classes=classes,
                                                                                            nt_samples=args.XAI_evaluate_nt_samples,   
                                                                                            n_steps=args.XAI_evaluate_n_steps,     
                                                                                            margin=0.1,    
                                                                                            topk=args.topk,
                                                                                            compare_sever_client_masks=0,
                                                                                            batch_size=args.XAI_evaluate_batch_size,
                                                                                            output_path=f"./res/epoch_{epoch}/",
                                                                                            verbose=0,
                                                                                            is_mode_6=True if args.mode in [6] else False)
            test_in_mask_acc_mean_list.append(in_mask_acc_mean)
            test_XAI_ACC_list.append(XAI_ACC)
            
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            is_best = 1
        if args.save_global == 1:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': global_weights,
                'train_accuracy': train_accuracy,
            }, is_best, local_idx=1000,is_global=1)
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            log.logger.debug(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            log.logger.debug(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Test Accuracy: {:.2f}% '.format(100*test_acc))
            log.logger.debug('Test Accuracy: {:.2f}% '.format(100*test_acc))
            print('Best Test Accuracy: {:.2f}% \n'.format(100*best_test_acc))
            log.logger.debug('Best Test Accuracy: {:.2f}% \n'.format(100*best_test_acc))
            print('Global XAI_ACC: {:.2f}% \n'.format(100 * XAI_ACC))
            log.logger.debug('Global XAI_ACC: {:.2f}% \n'.format(100*XAI_ACC))
            print('Global in_mask_acc_mean: {:.2f}% \n'.format(100 * in_mask_acc_mean))
            log.logger.debug('Global in_mask_acc_mean: {:.2f}% \n'.format(100*in_mask_acc_mean))
        scheduler.step()
        gc.collect()
        
        print("epoch Run Time: ",time.time()-epoch_start_time)
        print("test_loss_list.append",test_loss_list)
        print("test_acc_list.append",test_acc_list)
        print("test_in_mask_acc_mean.append",test_in_mask_acc_mean_list)
        print("test_XAI_ACC.append",test_XAI_ACC_list)

    # Test inference after completion of training
    test_acc, test_loss =  test_inference(args, global_model, test_dataset)
    test_loss_list.append(test_loss)

    print(f' \n Results after {args.epochs} global rounds of training:')
    log.logger.debug(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    log.logger.debug("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    if not os.path.isdir('save/objects'):
        os.makedirs('save/objects')
    file_name = 'save/objects/{}_{}_{}_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs,  args.iid,
               args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    log.logger.debug('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)

    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.show()
    # plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs,
    #                    args.iid, args.local_ep, args.local_bs))
    
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Train Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.show()
    # plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_train_acc.png'.
    #             format(args.dataset, args.model, args.epochs, 
    #                    args.iid, args.local_ep, args.local_bs))

    # # Plot Loss curve
    # plt.figure()
    # # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(test_loss_list)), test_loss_list, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.show()
    # plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs,
    #                    args.iid, args.local_ep, args.local_bs))
    # plt.figure()
    # # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(test_acc_list)), test_acc_list, color='k')
    # plt.ylabel('Average Test Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.show()
    # plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_test_acc.png'.
    #             format(args.dataset, args.model, args.epochs, 
    #                    args.iid, args.local_ep, args.local_bs))