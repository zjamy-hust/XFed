import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import math

# %matplotlib inline

import torch
import captum
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torchvision import models
from statistics import mean
import gc


from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

from numpy import ndarray
from typing import Any, Iterable, List, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.image as img
import torch.optim as optim
import torch.optim as optim

from options import args_parser
from models.models_resnet import ResNet18, ResNet18_with_mask
from localupdates.update import test_inference, DatasetSplit, generate_dataset_mask

def attribute_image_features(net, algorithm, input,truth, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=truth,
                                              **kwargs
                                             )
    return tensor_attributions

def XAI_evaluate_with_global_masks(local_model, 
                                   test_files_list, 
                                   path, 
                                   dataset_name,
                                   device, 
                                   XAI_labels, 
                                   classes, 
                                   nt_samples, 
                                   n_steps, 
                                   margin, 
                                   topk, 
                                   compare_sever_client_masks=0, 
                                   global_model=None, 
                                   batch_size=1,
                                   output_path="./",
                                   verbose=1,
                                   is_mode_6=False):
    """     重写XAI_evaluate，使其能够根据global model产生mask，从而实现global model和local model产生的mask进行对比
    
    与原始XAI_evaluate之间的差别：
        1、采用DataSplit加载测试样本

    Args:
        local_model (_type_): client model
        test_files_list (_type_): 人工挑选的测试样本列表，list。
        path (_type_): 测试样本的目录
        device (_type_): 
        XAI_labels (_type_): 测试样本的label index。
        classes (_type_): 测试样本的label名称。
        nt_samples
        n_steps
        margin： in_mask应当比out_mask大margin，防止噪声导致acc波动。
        topk
        compare_sever_client_masks (_type_): 是否进行client和global之间的对比
        global_model (_type_, optional): Server model. Defaults to None.
        batch_size: 测试和生成mask时的batch_size，为了计算的准确度，batch_size尽可能小，并且nt_samples和n_steps尽可能大。

    Returns:
        _type_: _description_
    """
    
    test_files_list_jpg = sorted(filter(lambda x: x.endswith(".jpg"), test_files_list))
    if len(test_files_list_jpg) <= 0:
        raise ValueError("test_files_list不包含jpg文件。")
    if len(test_files_list_jpg) != len(XAI_labels):
        raise ValueError("长度不匹配。")
    
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    
    torch.cuda.empty_cache()
    
    
    test_images = []
    npimg_list = []
    image_masks_by_human = []
    for im_idx in range(len(test_files_list_jpg)):
        im_name = test_files_list_jpg[im_idx]
        if verbose == 1:
            print(str(Path(path)/im_name))
        
        if dataset_name == "cifar10":
            dataset_size = 32
            normalize_mean = [0.4914, 0.4822, 0.4465]
            normalize_std = [0.2023, 0.1994, 0.2010]
            mode = torchvision.io.image.ImageReadMode.RGB
        elif dataset_name == "MNIST":
            dataset_size = 28
            normalize_mean = [0.1307] 
            normalize_std = [0.3081]
            mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            raise ValueError("dataset_name有误。")
        
        im_asset=read_image(str(Path(path)/im_name), mode=mode)
        original_image_asset = fn.resize(im_asset, size=[dataset_size,dataset_size])/255
        
        input_asset=torch.tensor(original_image_asset.unsqueeze(0).cpu().detach().numpy())
        input_asset.requires_grad = True
        
        input_asset_norm=fn.normalize(input_asset, mean=normalize_mean, std=normalize_std) #归一化，用来参与计算
        test_images.append((input_asset_norm.squeeze(dim=0),XAI_labels[im_idx]))
        
        npimg = original_image_asset.numpy()     # unnormalize，用来直接输出原图
        npimg_list.append(npimg)
        
        mask_im_name = im_name[:-4]+'_mask.png'
        if verbose == 1:
            print("mask_im_name",mask_im_name)
        truth_mask_tensor = read_image(str(Path(path)/mask_im_name),mode)
        truth_mask = cv2.imread(str(Path(path)/mask_im_name),cv2.IMREAD_GRAYSCALE)
        truth_mask_np=truth_mask_tensor.numpy()   
        
        image_masks_by_human.append((im_name, truth_mask_np, truth_mask))       #貌似truth_mask_np用不上？？？？？？？？？
        
    test_images_dataloader = DataLoader(DatasetSplit(test_images, [i for i in range(len(test_files_list_jpg))]), 
                                        batch_size=batch_size, 
                                        shuffle=False)
    
    XAI_inmask_list = []
    XAI_outmask_list = []
    i=0;
    correct=0;
    ig = IntegratedGradients(local_model)
    nt = NoiseTunnel(ig)
    for batch_idx, (images, labels, idxs) in enumerate(test_images_dataloader):
        input_asset_norm=images.to(device)
        examples_num = input_asset_norm.shape[0]
        
        if is_mode_6 == True:
            localmodel_mask = local_model.get_masks(input_asset_norm)
        output_asset = local_model(input_asset_norm)
        _, predicted_asset = torch.max(output_asset, 1)
        

        for i in range(examples_num):   #分别处理每一个样本
            if verbose == 1:
                print("predicted_asset",classes[predicted_asset[i]],"Truth", classes[labels[i]])
            example_index_in_all = batch_idx * test_images_dataloader.batch_size + i
            if verbose == 1:
                print("example index:", example_index_in_all)
            pixelnum_all=np.count_nonzero(image_masks_by_human[example_index_in_all][2])
            if verbose == 1:
                print("pixelnum_all", pixelnum_all)

            #the 2nd parameter can be input_asset or input_asset_norm, input_asset_norm will show better ACC in XAI
            attr_ig_nt = attribute_image_features(local_model,
                                                nt, 
                                                input_asset_norm[i].unsqueeze(0),
                                                truth=XAI_labels[int(image_masks_by_human[example_index_in_all][0][:-4])-1], 
                                                baselines=input_asset_norm[i].unsqueeze(0) * 0, 
                                                nt_type='smoothgrad_sq',  
                                                nt_samples=nt_samples, 
                                                n_steps=n_steps, 
                                                stdevs=0.2)
            attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            
            #计算二值化masks

            attr_combined = np.sum(attr_ig_nt, axis=2)/3
            # attr_combined = np.abs(attr_combined)
            attr_combined_flatten_sorted = np.sort(attr_combined.flatten())
            threshold_idx = math.ceil(topk * attr_combined_flatten_sorted.shape[0])
            threshold = attr_combined_flatten_sorted[attr_combined_flatten_sorted.shape[0] - threshold_idx]
            
            attr_hard_masks = (attr_combined >= threshold).astype(float)
            attr_soft_masks = (attr_combined-np.min(attr_combined))/(np.max(attr_combined)-np.min(attr_combined))

            truth_mask = image_masks_by_human[example_index_in_all][2]
            masked = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=truth_mask) 
            out_mask = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=255-truth_mask) 

            # masked = cv2.add(attr_soft_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=truth_mask) 
            # out_mask = cv2.add(attr_soft_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=255-truth_mask) 

            
            inmask_pixelnum=np.count_nonzero(masked)
            inmask_percent=inmask_pixelnum/pixelnum_all
            if verbose == 1:
                print("in mask pixelnum", inmask_pixelnum,pixelnum_all,inmask_percent)
            XAI_inmask_list.append(inmask_percent)
            out_pixelnum=np.count_nonzero(out_mask)
            outmask_percent=out_pixelnum/(32*32-pixelnum_all)
            if verbose == 1:
                print("out mask pixelnum", out_pixelnum,32*32-pixelnum_all, outmask_percent)
            XAI_outmask_list.append(outmask_percent)

            if inmask_percent > outmask_percent + margin:
                correct = correct+1
            
            torch.cuda.empty_cache()
            # orig_image = npimg_list[example_index_in_all]
            # fig, ((orig, mask, mask_with_orig, attr), (attr_with_orig, attr_mask, attr_mask_with_orig, attr_outmask)) = plt.subplots(2, 4)
            # orig.imshow(np.transpose(orig_image, (1, 2, 0)))
            # default_cmap = LinearSegmentedColormap.from_list(
            #     "RdWhGn", ["red", "white", "green"]
            # )
            # vmin, vmax = -1, 1
            # mask.imshow(truth_mask, cmap="Blues", vmin=0, vmax=1)
            # masks_with_orig = orig_image * (truth_mask>127).astype(float) + (np.ones_like(orig_image)*0.5) * (truth_mask<=127).astype(float)
            # mask_with_orig.imshow(np.transpose(masks_with_orig, (1, 2, 0)))
            # attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
            # attrs_with_orig = orig_image * (attr_hard_masks>0.5).astype(float) + (np.ones_like(orig_image)*0.5) * (attr_hard_masks<=0.5).astype(float)
            # attr_with_orig.imshow(np.transpose(attrs_with_orig, (1, 2, 0)),cmap=default_cmap,vmin=-1,vmax=1)
            # attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
            # attrs_with_orig = orig_image * (masked>0.5).astype(float) + (np.ones_like(orig_image)*0.5) * (masked<=0.5).astype(float)
            # attr_mask_with_orig.imshow(np.transpose(attrs_with_orig, (1, 2, 0)),cmap="Greens",vmin=0,vmax=1)
            
            # # attrs_with_orig_soft = orig_image * attr_soft_masks * (attr_soft_masks>0.3).astype(float) + (np.ones_like(orig_image)*0.5) * (masked<=0.3).astype(float)
            # # attr_outmask.imshow(np.transpose(attrs_with_orig_soft, (1, 2, 0)),cmap="Reds",vmin=0,vmax=1)
            
            # attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
            
            orig_image = npimg_list[example_index_in_all]
            fig, (orig, mask, attr, attr_in_mask, attr_outmask) = plt.subplots(1, 5)
            orig.axis('off')
            mask.axis('off')
            attr.axis('off')
            attr_in_mask.axis('off')
            attr_outmask.axis('off')
            orig.imshow(np.transpose(orig_image, (1, 2, 0)),cmap="gray")
            default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
            mask.imshow(truth_mask, cmap="Blues", vmin=0, vmax=1)
            attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
            attr_in_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
            
            attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
            
            fig.show()
            fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_result.png")
        
           
            plt.close()

            
    in_mask_acc_mean = mean(XAI_inmask_list)
    out_mask_acc_mean = mean(XAI_outmask_list)
    gc.collect()
    if verbose == 1:
        print("in_mask_acc_mean",in_mask_acc_mean,"out_mask_acc_mean",out_mask_acc_mean,"XAI ACC", correct/len(XAI_labels))
    return in_mask_acc_mean,out_mask_acc_mean,correct/len(XAI_labels)

if __name__=="__main__":
    model_path = "mnist_test.pth.tar"
    args = args_parser()
    args.gpu=1
    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu' 

    asset_path='assets'
    
    args.dataset = "MNIST"
    if args.dataset == "cifar10":
        num_classes = 10
        input_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
        #            'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7, 'ship' 8, 'truck' 9)
        XAI_labels=[7, 8, 2, 2, 0, 5, 7, 9, 2, 8, 8, 2, 8, 2, 5, 8, 0, 7, 5, 5,1,1,3,3,4,4,6,6,9,3 ]
        assetpath = str(Path(asset_path)/'cifar_asset')
        print("assetpath",assetpath)
        files = os.listdir(assetpath)        
    elif args.dataset == "MNIST":
        num_classes = 10
        input_channel = 1
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        assetpath = str(Path(asset_path)/'mnist_asset')
        with open(assetpath+"/labels.txt",'r') as f:
            read_res = f.readlines()
            XAI_labels = [int(line.strip()) for line in read_res]
        print("XAI labels", XAI_labels)
        print("assetpath",assetpath)
        files = os.listdir(assetpath)        
    

    #To prepare network
    torch.cuda.set_device(args.gpu)
    args.mode = 1
    args.output_hard_mask = 1
    if args.mode in [6,7,8]:
        net = ResNet18_with_mask(num_classes, input_channel,True if args.output_hard_mask==1 else False)
    else:
        net = ResNet18(num_classes, input_channel)
    # checkpoint = torch.load(model_path)
    # net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    a,b,c=XAI_evaluate_with_global_masks(net,
                                        files,
                                        assetpath,
                                        dataset_name=args.dataset,
                                        device=device,
                                        XAI_labels=XAI_labels,
                                        classes=classes,
                                        nt_samples=5,   #测试数值
                                        n_steps=5,      #测试数值
                                        margin=0.1,     #in_mask和out_mask之间的差距
                                        topk=0.07,
                                        compare_sever_client_masks=0,
                                        batch_size=32,
                                        output_path=f"./res/epoch_-1/",
                                        verbose=0,
                                        is_mode_6=False)
    
    print("a",a,"b",b,"c",c)
