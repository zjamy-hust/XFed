#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import argparse
def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=16,
                        help="number of users: K")
    parser.add_argument('--client_selection_num', type=int, default=16,
                        help='选中client的数量。')
    parser.add_argument('--local_ep', type=int, default=50,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--is_aggregate_with_weights', type=int, default=0,
                        help="聚合时，是否采用客户端的样本作为权重。")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--stopping_rounds', type=int, default=20,
                        help='rounds of early stopping')


    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='test, restnet18, shufflenetv2')
    parser.add_argument('--save_local', type=int, default=0, help='save local models or not')
    parser.add_argument('--save_global', type=int, default=1, help='save global models or not')

    
    # other arguments
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="name of dataset. ['cifar10','cifar100','MNIST']")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--comparing_shared', type=int, default=0, help='comparing shared or not')
    parser.add_argument('--shared_data', type=float, default=0, help='using shared data or not')
    parser.add_argument('--partition', type=str, default='hetero-dir', help='homo, hetero-dir ')
    parser.add_argument('--compare_sever_client_masks', type=int, default=1, help='XAI评估时，是否对比global和local生成的mask。')
    
    parser.add_argument('--XAI_evaluate_batch_size', type=int, default=16,help="XAI_evaluate过程中，一次计算的样本数量。")
    parser.add_argument('--XAI_evaluate_nt_samples', type=int, default=5,help="XAI_evaluate过程中，可解释算法采样的数量。")
    parser.add_argument('--XAI_evaluate_n_steps', type=int, default=5,help="XAI_evaluate过程中，，一个样本step数量。")
    parser.add_argument('--mode', type=int, default=6,
                        help='[0,1,2,3,4,5,6,7,8]。0表示FedAvg，随机选择客户端； \
                                    1表示data augmentation训练，训练前，利用server model和trainning dataset生成mask，训练时进行一次原始样本训练+一次带mask训练，测试时计算acc+XAI Acc（暂时如此）；\
                                    2表示仅测试时采用mask，然后通过acc筛选客户端，测试前会通过server model和testing dataset产生mask；\
                                    3表示采用带mask的图片masked_image进行训练，masked_image和image的预测结果之差作为正则化项，测试时计算acc+XAI Acc（暂时如此）；\
                                    4表示fedProx; \
                                    5表示以任务acc为参考标准选取最优client（oort'）；)
    
    #mode == 1需要关注的参数
    parser.add_argument('--mode1_start_epoch', type=int, default=1,help="data augmentation需要跳过前面几轮再执行。")
    parser.add_argument('--train_mask_batch_size', type=int, default=32,help="生成训练集mask，一次计算的mask数量。")
    parser.add_argument('--train_mask_nt_samples', type=int, default=2,help="IG算法，一个样本采样的次数。")
    parser.add_argument('--train_mask_n_steps', type=int, default=1,help="IG算法，一个样本step数量。")
    
    #mode == 2需要关注的参数
    parser.add_argument('--mode2_end_epoch', type=int, default=5,help="mode 2需要在第几轮之后停止使用。")
    parser.add_argument('--test_mask_batch_size', type=int, default=2000,help="生成测试集mask，一次计算的mask数量。")
    parser.add_argument('--test_mask_nt_samples', type=int, default=1,help="IG算法，一个样本采样的次数。")
    parser.add_argument('--test_mask_n_steps', type=int, default=1,help="IG算法，一个样本step数量。")
     
    #mode == 3需要关注的参数 
    # parser.add_argument('--mse_loss_lambda', type=float, default=1,help="mode 3中两个预测结果近似程度正则化项的系数。")
    parser.add_argument('--mode3_train_mask_batch_size', type=int, default=1000,help="生成训练集mask，一次计算的mask数量。")
    parser.add_argument('--mode3_train_mask_nt_samples', type=int, default=1,help="IG算法，一个样本采样的次数。")
    parser.add_argument('--mode3_train_mask_n_steps', type=int, default=1,help="IG算法，一个样本step数量。")
     
    #mode == 4需要关注的参数 
    # parser.add_argument('--weights_regularization_lambda', type=float, default=1,help="mode 4中权重正则化项的系数。")
    
    #mode == 3 4 需要关注的参数 
    parser.add_argument('--mse_loss_lambda', type=float, default=0.01,help="")
    parser.add_argument('--mapping', type=int, default=0,help="")
    parser.add_argument('--topk', type=float, default=0.3,help="统计人工标注的mask与原图总像素比例之后，推荐MNIST采用0.1。")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    
    parser.add_argument('--random_seed', type=int, default=40212202)
    parser.add_argument('--gpu', default=0, type=int, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    #mode == 6 需要关注的参数 
    parser.add_argument('--output_hard_mask', type=int, default=1,help="是否将attention mask转为hard mask。方式是gumble-softmax。")

    args = parser.parse_args()
    return args
