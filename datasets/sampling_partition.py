import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100, MNIST
from torchvision.datasets import DatasetFolder
from torchvision import datasets, transforms
import torch.utils.data as data
import math
import copy
import time

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int((len(dataset))/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int((len(dataset))/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int((len(dataset))/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

#to be changed
def cifar_noniid(dataset, num_users, partition):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    alpha=0.5
    datadir = './data/'
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = int(X_train.shape[0])
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        idx_batch = np.array_split(idxs, num_users)
        net_dataidx_map = {i: idx_batch[i] for i in range(num_users)}
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = int(X_train.shape[0])
        print('x.shape',X_train.shape[0],' y.shape',y_train.shape[0])
        y_train = y_train[:] # 截取训练集
        print('new shape',y_train.shape[0])
        net_dataidx_map = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map

def cifar100_noniid(dataset, num_users, partition):
    """
    Sample non-I.I.D client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    alpha=0.5
    datadir = './data/'
    X_train, y_train, _, _ = load_cifar100_data(datadir)
    n_train = int(X_train.shape[0])

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        idx_batch = np.array_split(idxs, num_users)
        net_dataidx_map = {i: idx_batch[i] for i in range(num_users)}
    elif partition == "hetero-dir":
        min_size = 0
        K = 100
        N = int(X_train.shape[0])
        print('x.shape',X_train.shape[0],' y.shape',y_train.shape[0])
        y_train = y_train[:] # 截取训练集
        print('new shape',y_train.shape[0])
        net_dataidx_map = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
#     print('net_dataidx_map',net_dataidx_map)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    #return y_train, net_dataidx_map, traindata_cls_counts
    return net_dataidx_map

def mnist_noniid(dataset, num_users, partition):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    alpha=0.5
    datadir = './data/'
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = int(X_train.shape[0])
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        idx_batch = np.array_split(idxs, num_users)
        net_dataidx_map = {i: idx_batch[i] for i in range(num_users)}
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = int(X_train.shape[0])
        print('x.shape',X_train.shape[0],' y.shape',y_train.shape[0])
        y_train = y_train[:] # 截取训练集
        print('new shape',y_train.shape[0])
        net_dataidx_map = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


#to be changed end

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                       transform=train_transform)
    train_dataset_cifar100 = datasets.CIFAR10('../data', train=True, download=True,
                                       transform=train_transform)
    user_groups = cifar_iid(train_dataset, 16)
    user_groups = cifar_noniid(train_dataset, 16,'hetero-dir')
    user_groups = cifar100_iid(train_dataset_cifar100, 16)
    user_groups = cifar100_noniid(train_dataset_cifar100, 16,'hetero-dir')
    # for j in range(16):
    #     print('user_groups[',j,']',len(user_groups[j]))
    # partition_data('cifar10', '../data', 'homo', 16, 0.5)
    # partition_data('cifar100', '../data', 'homo', 16, 0.5)
