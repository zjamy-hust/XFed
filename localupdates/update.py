#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from utils import weights_norm_2L_regularization

best_global_acc = 0

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
#        return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label), self.idxs[item]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.valloader = self.train_val_test(dataset, list(idxs))
        self.device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_local_acc = 0
        #self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        # idxs_train = idxs[:int(len(idxs))]            #这里应该和FedAvg对齐？？？？？？？？？？？？？
        idxs_val = idxs[int(0.9*len(idxs)):] # is it iid? needs improve



        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        valloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=128, shuffle=False)

        return trainloader, valloader
    
    def update_weights(self, model, global_round, user):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
            
        loss_epochs=0
        for iter in range(self.args.local_ep):
            is_best = 0
            batch_loss = []
            model.train()
            
            for batch_idx, (images, labels, idxs) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(len(images), len(labels))
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar(f'user{user}_train_loss', loss.item(),loss_epochs)
                batch_loss.append(loss.item())
                optimizer.step()
                loss_epochs+=1
                
            if self.args.verbose :
                print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.inference(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss), best_model

    def inference(self, model, global_round=1000,user=1000):

        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct, is_best = 0.0, 0.0, 0.0, 1

        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.valloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
    
                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
    
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
            # Save checkpoint.
        if accuracy > self.best_local_acc:
            # print('Saving..')
            # state = {'model_state_dict':model.state_dict(),
            #         'loss':loss
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_local.pth')
            self.best_local_acc = accuracy
            is_best = 1
        if self.args.verbose :
            print(f'Global:{global_round}, user:{user}, train accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best local acc: {100*self.best_local_acc:.2f}%')
        return accuracy, loss, is_best

    def test_inference_with_global_mask(self, model, global_round=1000,user=1000):

        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct, is_best = 0.0, 0.0, 0.0, 1

        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.valloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
    
                # Inference
                outputs = model(images)     #不加mask，直接用原图进行测试
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
    
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
            # Save checkpoint.
        if accuracy > self.best_local_acc:
            # print('Saving..')
            # state = {'model_state_dict':model.state_dict(),
            #         'loss':loss
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_local.pth')
            self.best_local_acc = accuracy
            is_best = 1
        if self.args.verbose :
            print(f'Global:{global_round}, user:{user}, train accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best local acc: {100*self.best_local_acc:.2f}%')
        return accuracy, loss, is_best
    
    def update_weights_fedprox(self, model, global_round, user, global_model):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
        
        loss_epochs=0
        for iter in range(self.args.local_ep):
            is_best = 0
            a_loss_max,b_loss_max = 1,1
            batch_loss = []
            model.train()
            
            for batch_idx, (images, labels, idxs) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(len(images), len(labels))
                model.zero_grad()
                log_probs = model(images)
                # loss = self.criterion(log_probs, labels) + self.args.weights_regularization_lambda * weights_norm_2L_regularization(global_model.state_dict(), model.state_dict())
                a_loss = self.criterion(log_probs, labels)
                b_loss = weights_norm_2L_regularization(global_model.state_dict(), model.state_dict())
                if a_loss.item() > a_loss_max:
                    a_loss_max = a_loss.item()
                if b_loss.item() > b_loss_max:
                    b_loss_max = b_loss.item()
                loss = 0.5*a_loss + 0.5*b_loss*(a_loss_max/b_loss_max)
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar(f'user{user}_train_loss', loss.item(),loss_epochs)
                batch_loss.append(loss.item())
                optimizer.step()
                
                loss_epochs+=1
                
            if self.args.verbose :
                print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.inference(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss), best_model

    def update_weights_augmentation(self, model, global_round, user, train_masks):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
            
        loss_epochs=0
        for iter in range(self.args.local_ep):
            is_best = 0
            batch_loss = []
            model.train()
            
            for batch_idx, (images, labels, idxs) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # # print(len(images), len(labels))
                # model.zero_grad()
                # log_probs = model(images)
                # loss = self.criterion(log_probs, labels)
                # # optimizer.zero_grad()
                # loss.backward(retain_graph=True)
               
                # self.logger.add_scalar('loss', loss.item())
                # batch_loss.append(loss.item())
                # optimizer.step()
                
                #获取训练样本对应的masks
                masks_with_idx = [train_masks[int(index)] for index in idxs]  #根据idxs索引对应的mask
                masks = torch.stack([item[0] for item in masks_with_idx],dim=0).unsqueeze(1)
                images_ = masks * images
                # print(len(images), len(labels))
                model.zero_grad()
                log_probs = model(images_)
                loss = self.criterion(log_probs, labels)
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar(f'user{user}_train_loss', loss.item(),loss_epochs)
                batch_loss.append(loss.item())
                optimizer.step()
                
                loss_epochs+=1
                
            if self.args.verbose :
                print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.inference(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss), best_model

    def update_weights_augmentation_similarity(self, model, device, global_round, user, train_mask_batch_size, train_mask_nt_samples, train_mask_n_steps, topk, mse_loss_lambda,mapping):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
            
        loss_epochs=0
        for iter in range(self.args.local_ep):
            is_best = 0
            a_loss_max,b_loss_max = 1,1
            batch_loss = []
            model.train()
            
            #根据最新的模型产生mask
            cur_train_masks = generate_dataset_mask(model,
                                                    dataset=self.trainloader.dataset.dataset,
                                                    idxs=self.trainloader.dataset.idxs,
                                                    batch_size=train_mask_batch_size,
                                                    nt_samples=train_mask_nt_samples,
                                                    n_steps=train_mask_n_steps,
                                                    device=device,
                                                    topk = topk)          #速度比较慢、占用空间比较大？？？？？？？？？？？？考虑是否吸收到每一个batch中，即使生成？
            
            for batch_idx, (images, labels, idxs) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                #获取训练样本对应的masks，先进行带mask的前向过程
                masks_with_idx = [cur_train_masks[int(index)] for index in idxs]  #根据idxs索引对应的mask
                masks = torch.stack([item[0] for item in masks_with_idx],dim=0).unsqueeze(1)
                images_ = masks * images
                # print(len(images), len(labels))
                model.zero_grad()
                log_probs_with_masks = model(images_)
                
                #进行不带mask的前向过程
                log_probs = model(images)
                a_loss = self.criterion(log_probs, labels)
                b_loss = F.mse_loss(log_probs_with_masks,log_probs.detach())
                if a_loss.item() > a_loss_max:
                    a_loss_max = a_loss.item()
                if b_loss.item() > b_loss_max:
                    b_loss_max = b_loss.item()
                if ( mapping == 1) :
                    rate = (a_loss_max/b_loss_max)
                else:
                    rate = mse_loss_lambda

                loss = a_loss + rate * b_loss   #只回传一次梯度。
                
                # loss = self.criterion(log_probs_with_masks, labels) + mse_loss_lambda * F.mse_loss(log_probs_with_masks,log_probs.detach())   #只回传一次梯度。
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar(f'user{user}_train_loss', loss.item(),loss_epochs)
                batch_loss.append(loss.item())
                optimizer.step()
                
                loss_epochs+=1
                
                
            if self.args.verbose :
                print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.inference(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss), best_model

    def update_weights_with_global_mask(self, model, global_round, user):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
            
        loss_epochs=0
        for iter in range(self.args.local_ep):
            is_best = 0
            batch_loss = []
            model.train()
            
            for batch_idx, (images, labels, idxs) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(len(images), len(labels))
                model.zero_grad()
                mask = model.get_masks(images)
                images_ = images * mask[:,:,:,:,1]
                log_probs = model(images_)
                loss = self.criterion(log_probs, labels) + torch.mean(mask[:,:,:,:,1])
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar(f'user{user}_train_loss', loss.item(),loss_epochs)
                batch_loss.append(loss.item())
                optimizer.step()
                loss_epochs+=1
                
            if self.args.verbose :
                print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.test_inference_with_global_mask(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss), best_model
    
def generate_dataset_mask(local_init_model, dataset, idxs, batch_size, nt_samples, n_steps, device, topk):   
    """
    利用服务器发来的模型产生mask
    
    Args:
        local_init_model (_type_): 服务器发来的模型
        dataset (_type_): 需要生成mask的数据集
        idxs (_type_): idx集合
        batch_size (_type_): batch_size
        nt_samples (_type_): smoothGrad方法的采样次数
        topk (float, optional): 选取前topk百分比的显著图分数. Defaults to 0.5.
    """
    ig = IntegratedGradients(local_init_model)
    nt = NoiseTunnel(ig)

    #构建dataloader
    dataloader_for_masks = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=False)    #无论是train、还是test，都不进行shuffle
    
    masks={}
    for batch_idx, (images, labels, idxs) in enumerate(dataloader_for_masks):
        images, labels = images.to(device), labels.to(device)
        local_init_model.zero_grad()
        tensor_attributions = nt.attribute(images,
                                            target=labels,
                                            baselines=images * 0, 
                                            nt_type='smoothgrad_sq',  
                                            nt_samples=nt_samples, 
                                            n_steps=n_steps,
                                            stdevs=0.2    #nt_samples的取值，还需要调参试一下
                                            ).permute(0,2,3,1)        #产生的归因值是梯度，而非deeplift那种近似分数
        # tensor_attributions_unbind=torch.unbind(tensor_attributions,dim=0)
        for i in range(tensor_attributions.shape[0]):
            tensor_attributions_idx = tensor_attributions[i]
            tensor_attributions_idx = torch.sum(tensor_attributions_idx,dim=-1)/3
            tensor_attributions_idx_flatten, _ = tensor_attributions_idx.flatten().sort()
            threshold_idx = math.ceil(topk * tensor_attributions_idx_flatten.shape[0])
            tensor_attributions_threshold = tensor_attributions_idx_flatten[tensor_attributions_idx_flatten.shape[0] - threshold_idx]
            attributions_masks = (tensor_attributions_idx >= tensor_attributions_threshold).float()
            #expension+filter (cv2?)?????????
            
            masks[int(idxs[i])]=[attributions_masks,images[i]]    #保存三个变量，idxs用于根据dataset索引mask，attributions_masks表示mask，images表示原始图片，用于比对索引是否正确。
            
        torch.cuda.empty_cache()#2100->1700M
        
    return masks

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    global best_global_acc

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
    # print("device",device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset,[i for i in range(len(test_dataset))]), batch_size=128,
                            shuffle=False)
    batch_num = 0
    with torch.no_grad():
        for batch_idx, (images, labels, idxs) in enumerate(testloader):
            batch_num += 1
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
    
            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
    
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct/total
    loss = loss/batch_num
    if accuracy > best_global_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_global.pth')
            best_global_acc = accuracy
    print(f'test-full accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best_global_acc:{100*best_global_acc:.2f}%')
    return accuracy, loss

def test_inference_with_mask(args, model, test_dataset, test_masks):
    """ Returns the test accuracy and loss.
    """
    global best_global_acc

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
    # print("device",device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset,[i for i in range(len(test_dataset))]), batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels, idxs) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            
            #获取训练样本对应的masks
            masks_with_idx = [test_masks[int(index)] for index in idxs]  #根据idxs索引对应的mask
            masks = torch.stack([item[0] for item in masks_with_idx],dim=0).unsqueeze(1)
            images_ = masks * images
    
            # Inference
            outputs = model(images_)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
    
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct/total
    if accuracy > best_global_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_global.pth')
            best_global_acc = accuracy
    print(f'test-full accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best_global_acc:{100*best_global_acc:.2f}%')
    return accuracy, loss

def test_inference_with_global_mask(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    global best_global_acc

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
    # print("device",device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset,[i for i in range(len(test_dataset))]), batch_size=128,
                            shuffle=False)
    batch_num = 0
    with torch.no_grad():
        for batch_idx, (images, labels, idxs) in enumerate(testloader):
            batch_num += 1
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
    
            # Inference
            outputs = model(images)     #直接采用原图进行测试
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
    
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct/total
    loss = loss/batch_num
    if accuracy > best_global_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_global.pth')
            best_global_acc = accuracy
    print(f'test-full accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best_global_acc:{100*best_global_acc:.2f}%')
    return accuracy, loss
