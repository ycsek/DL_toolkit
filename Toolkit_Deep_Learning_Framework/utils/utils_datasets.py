'''
Author: Jason Shi
Date: 01-11-2024 15:53:22
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 10-02-2025 10:57:05
'''

#! This module is responsible for loading and preprocessing different datasets.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import requests
import zipfile
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def get_dataset(dataset, data_path):
    '''
    @Description: This module is responsible for loading and preprocessing different datasets, including MNIST, CIFAR-10, CIFAR-100, Tiny-imagenet and SVHN.

    @param:
    name(str): The name of the dataset, including 'MNIST', 'CIFAR-10', 'CIFAR-100', 'Tiny-imagenet' and 'SVHN'.
    batch_size(int): The batch size of the dataset.
    num_workers(int): The number of workers for data loading (Number of sub-processes used to load data)

    @return:
    train_loader, test_loader: The training and testing data loaders.

    '''

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Data preprocessing
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(
            data_path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(
            data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'Tiny':
        if not os.path.exists(os.path.join(data_path, "tiny-imagenet-200")):
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # 248MB
            print("Downloading Tiny-ImageNet")
            r = requests.get(url, stream=True)
            with open(os.path.join(data_path, "tiny-imagenet-200.zip"), "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            print("Unziping Tiny-ImageNet")
            with zipfile.ZipFile(os.path.join(data_path, "tiny-imagenet-200.zip")) as zf:
                zf.extractall(path=data_path)

        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.ImageFolder(root=os.path.join(
            data_path, 'tiny-imagenet-200/train'), transform=transform)
        dst_test = datasets.ImageFolder(root=os.path.join(
            data_path, 'tiny-imagenet-200/val'), transform=transform)

        images_all = []
        for i in range(len(dst_test)):
            images_all.append(torch.unsqueeze(dst_test[i][0], dim=0))
        images_all = torch.cat(images_all, dim=0).to("cpu")

        df = pd.read_csv(os.path.join(data_path, 'tiny-imagenet-200',
                         'val', 'val_annotations.txt'), sep='\t', header=None)
        img_names = [x[0].split('/')[-1] for x in dst_test.samples]
        labels_all = [df[df[0] == x][1].values[0] for x in img_names]
        labels_all = [dst_train.class_to_idx[x] for x in labels_all]
        labels_all = torch.tensor(labels_all, dtype=torch.long)

        dst_test = TensorDataset(images_all, labels_all)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(
            data_path, split='train', download=True, transform=transform)
        dst_test = datasets.SVHN(
            data_path, split='test', download=True, transform=transform)

    else:
        exit('unknown dataset: %s' % dataset)

    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=256, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=128, shuffle=True, num_workers=2)

    return trainloader, testloader, channel, im_size, num_classes
