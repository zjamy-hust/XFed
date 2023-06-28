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
from localupdates.update import test_inference, DatasetSplit, generate_dataset_mask
from options import args_parser
import copy
args = args_parser()
args.gpu=1
device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
# showimg=1
run=1


# model_path="save_checkpoints/xai_analysis/global.iid1.16_15.pth.tar"
# model_path="save_checkpoints/resnet18_global/global.iid0.gpu0.mode5.global_epoch20.local_ep5.num_users3.ckpt.pth.tar"
asset_path='assets'


# def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
#     # given values should be non-negative
#     assert percentile >= 0 and percentile <= 100, (
#         "Percentile for thresholding must be " "between 0 and 100 inclusive."
#     )
#     sorted_vals = np.sort(values.flatten())
#     cum_sums = np.cumsum(sorted_vals)
#     threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
#     return sorted_vals[threshold_id]

def attribute_image_features(net, algorithm, input,truth, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=truth,
                                              **kwargs
                                             )
    return tensor_attributions


def imshow(img, transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
    
    test_files_list_jpg = sorted(filter(lambda x: x.endswith(".jpg"), test_files_list))
    if len(test_files_list_jpg) <= 0:
        raise ValueError("test_files_list does not contain jpg files.")
    if len(test_files_list_jpg) != len(XAI_labels):
        raise ValueError("The lengths do not match.")
    
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
        
        input_asset_norm=fn.normalize(input_asset, mean=normalize_mean, std=normalize_std)
        test_images.append((input_asset_norm.squeeze(dim=0),XAI_labels[im_idx]))
        
        npimg = original_image_asset.numpy()     
        npimg_list.append(npimg)
        
        mask_im_name = im_name[:-4]+'_mask.png'
        if verbose == 1:
            print("mask_im_name",mask_im_name)
        truth_mask_tensor = read_image(str(Path(path)/mask_im_name),mode)
        truth_mask = cv2.imread(str(Path(path)/mask_im_name),cv2.IMREAD_GRAYSCALE)
        truth_mask_np=truth_mask_tensor.numpy()   
        
        image_masks_by_human.append((im_name, truth_mask_np, truth_mask)) 
        
    test_images_dataloader = DataLoader(DatasetSplit(test_images, [i for i in range(len(test_files_list_jpg))]), 
                                        batch_size=batch_size, 
                                        shuffle=False)
    
    if compare_sever_client_masks == 1:
        test_images_masks_by_local_model = generate_dataset_mask(local_model, 
                                                            test_images, 
                                                            [i for i in range(len(test_files_list_jpg))], 
                                                            batch_size, 
                                                            nt_samples, 
                                                            n_steps, 
                                                            device, 
                                                            topk)
        torch.cuda.empty_cache()
        test_images_masks_by_global_model = generate_dataset_mask(global_model, 
                                                                test_images, 
                                                                [i for i in range(len(test_files_list_jpg))], 
                                                                batch_size, 
                                                                nt_samples, 
                                                                n_steps, 
                                                                device, 
                                                                topk)
        torch.cuda.empty_cache()
    
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
        

        for i in range(examples_num):   
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

            truth_mask = image_masks_by_human[example_index_in_all][2]
            masked = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=truth_mask) 
            out_mask = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=255-truth_mask) 

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
            fig, (orig, mask, attr, attr_mask, attr_outmask) = plt.subplots(1, 5)
            orig.axis('off')
            mask.axis('off')
            attr.axis('off')
            attr_mask.axis('off')
            attr_outmask.axis('off')
            orig.imshow(np.transpose(npimg_list[example_index_in_all], (1, 2, 0)))
            default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
            mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
            attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
            attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
            attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
            fig.show()
            fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_result.png")
            
            if is_mode_6 == True:
                fig, (orig, localmodel_mask_) = plt.subplots(1, 2)
                orig.imshow(np.transpose(npimg_list[example_index_in_all], (1, 2, 0)),cmap=default_cmap,vmin=-1,vmax=1)
                localmodel_mask_.imshow(localmodel_mask[i][0,:,:,1].cpu().data.numpy(),cmap=default_cmap,vmin=-1,vmax=1)
                fig.colorbar(plt.cm.ScalarMappable(cmap=default_cmap,norm=Normalize(vmin=-1., vmax=1.)))
                fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_local_model_mask.jpg")
            
            #显示globale和local的mask
            if compare_sever_client_masks == 1:
                fig, (orig, local_mask_, global_mask_,localmodel_mask_) = plt.subplots(1, 4)
                orig.imshow(np.transpose(npimg_list[example_index_in_all], (1, 2, 0)))
                local_mask_.imshow(test_images_masks_by_local_model[example_index_in_all][0].cpu().data.numpy())
                global_mask_.imshow(test_images_masks_by_global_model[example_index_in_all][0].cpu().data.numpy())
                localmodel_mask_.imshow(localmodel_mask[example_index_in_all][0,:,:,1].cpu().data.numpy())
                fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_compare_global_local.jpg")
            plt.close()

            
    in_mask_acc_mean = mean(XAI_inmask_list)
    out_mask_acc_mean = mean(XAI_outmask_list)
    gc.collect()
    if verbose == 1:
        print("in_mask_acc_mean",in_mask_acc_mean,"out_mask_acc_mean",out_mask_acc_mean,"XAI ACC", correct/len(XAI_labels))
    return in_mask_acc_mean,out_mask_acc_mean,correct/len(XAI_labels)

if __name__=="__main__":
    from models.models_resnet import ResNet18

    asset_path='assets'
    # classes = ('plane', 'car', 'bird', 'cat',
    #     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # # classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
    # #            'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7, 'ship' 8, 'truck' 9)
    # XAI_labels=[7, 8, 2, 2, 0, 5, 7, 9, 2, 8, 8, 2, 8, 2, 5, 8, 0, 7, 5, 5,1,1,3,3,4,4,6,6,9,3 ]
    # assetpath = str(Path(asset_path)/'cifar_asset')
    # print("assetpath",assetpath)
    # files = os.listdir(assetpath)        
    
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
    num_classes = 10
    input_channel = 3
    net = ResNet18(num_classes, input_channel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # checkpoint = torch.load(model_path)
    # net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    # a,b,c = XAI_evaluate(net,files,assetpath,1,1,device=device,XAI_labels=XAI_labels, classes=classes)
    a,b,c=XAI_evaluate_with_global_masks(net,
                                        files,
                                        assetpath,
                                        dataset_name="MNIST",
                                        device=device,
                                        XAI_labels=XAI_labels,
                                        classes=classes,
                                        nt_samples=5,   
                                        n_steps=5,      
                                        margin=0.1,     
                                        topk=0.1,
                                        compare_sever_client_masks=0,
                                        batch_size=32,
                                        output_path=f"./res/epoch_-1/",
                                        verbose=0,
                                        is_mode_6=False)
    
    print("a",a,"b",b,"c",c)
