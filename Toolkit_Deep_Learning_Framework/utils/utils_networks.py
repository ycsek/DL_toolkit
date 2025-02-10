'''
Author: Jason Shi
Date: 01-11-2024 15:53:22
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 03-11-2024 23:21:12
'''

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation


#! This module defines different neural network architectures.
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP


class MLP(nn.Module):
    def __init__(self, channel, num_classes, res=32):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel == 1 else res*res*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

# ConvNet


class ConvNet(nn.Module):
    def __init__(self, num_classes=10, input_size=(3, 32, 32)):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            out = self.layer1(dummy_input)
            out = self.layer2(out)
            self.flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# LeNet


class LeNet(nn.Module):
    def __init__(self, channel, num_classes, res=32):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel ==
                      1 else 0, stride=1 if res == 32 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x

# AlexNet


class AlexNet(nn.Module):
    def __init__(self, channel, num_classes, res=32):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1 if res ==
                      32 else 2, padding=4 if channel == 1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# VGG
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm', res=32):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg[vgg_name], norm, res)
        self.classifier = nn.Linear(
            512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm, res):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                           nn.GroupNorm(
                               x, x, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1 if res == 32 else 2)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)


def VGG11_Tiny(channel, num_classes):
    return VGG('VGG11', channel, num_classes, res=64)


def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')


def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)


def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)


def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)

# ResNet

# ResNet


class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)  # modification
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)  # modification
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(
            64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel ==
                                    1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm', res=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm
        if res == 64:
            self.conv1 = nn.Conv2d(
                channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(
            64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(
            64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18BN_Tiny(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm', res=64)


def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


def ResNet18_Tiny(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, res=64)


def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes)


def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes)


def ResNet18ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


def ResNet6ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [1, 1, 1, 1], channel=channel, num_classes=num_classes)

#  Returns the corresponding network instance by name


def get_network(name, num_classes, channel, input_size=(32, 32), dist=True):
    '''
    @param:
    name(str): the name of the network
    channel(int): image channel
    num_classes(int): the number of classes
    input_size(tuple): the size of the input image
    dist(bool): whether to use distributed training

    @return:
    net(nn.Module): the network instance
    '''

    if name == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif name == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes,
                      input_size=input_size)
    elif name == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif name == 'alexnet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif name == 'VGG11':
        name = VGG11(num_classes=num_classes)
    elif name == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif name == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif name == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif name == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif name == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes)

    elif name == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, )
    elif name == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes,)

    elif name == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes)

    elif name == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes,)

    elif name == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, )
    elif name == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, )

    elif name == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes,)
    elif name == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes,)

    else:
        net = None
        exit('Error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num > 0:
            device = 'cuda'
            if gpu_num > 1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net
