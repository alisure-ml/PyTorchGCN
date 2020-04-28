import os
import cv2
import dgl
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
from layers.gat_layer import GATLayer
from layers.gcn_layer import GCNLayer
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from layers.mlp_readout_layer import MLPReadout
from torch.utils.data import Dataset, DataLoader
from layers.gated_gcn_layer import GatedGCNLayer
from layers.graphsage_layer import GraphSageLayer
from visual_embedding_2_norm import DealSuperPixel


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        Tools.print()
        Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        Tools.print()
        Tools.print('Cuda not available')
        device = torch.device("cpu")
    return device


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path
        # self.transform = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
        #                                      transforms.RandomHorizontalFlip()]) if self.is_train else None

        self.tran_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.tran_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train,
                                         transform=self.tran_train if self.is_train else self.tran_test)

        # 3. Super Pixel
        self.image_size = image_size
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set.__getitem__(idx)

    pass


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, padding=1, ks=3, has_relu=True, has_bn=True, bias=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    pass


class CNNNet(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = False
        conv_stride = 4
        avg_range = 3

        self.conv0 = ConvBlock(3, 64, stride=conv_stride, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        # self.conv5 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool5 = nn.AvgPool2d(3, 1, padding=1)
        # self.conv6 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool6 = nn.AvgPool2d(3, 1, padding=1)
        # self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(128, 10)
        pass

    def forward(self, x):
        e = self.conv0(x)

        e = self.conv1(e)
        e = self.pool1(e)
        e = self.conv2(e)
        e = self.pool2(e)
        e = self.pool_m1(e)

        e = self.conv3(e)
        e = self.pool3(e)
        e = self.conv4(e)
        e = self.pool4(e)
        e = self.pool_m2(e)

        # e = self.conv5(e)
        # e = self.pool5(e)
        # e = self.conv6(e)
        # e = self.pool6(e)
        # e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet2(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True
        conv_stride = 2
        avg_range = 3

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv03 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv04 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv05 = ConvBlock(64, 64, stride=conv_stride, padding=0, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        # self.conv5 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool5 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        # self.conv6 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool6 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        # self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(128, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)
        # e = self.conv03(e)
        # e = self.conv04(e)
        e = self.conv05(e)

        e = self.conv1(e)
        e = self.pool1(e)
        e = self.conv2(e)
        e = self.pool2(e)
        e = self.pool_m1(e)

        e = self.conv3(e)
        e = self.pool3(e)
        e = self.conv4(e)
        e = self.pool4(e)
        e = self.pool_m2(e)

        # e = self.conv5(e)
        # e = self.pool5(e)
        # e = self.conv6(e)
        # e = self.pool6(e)
        # e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet22(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True
        conv_stride = 4
        avg_range = 3

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 128, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv03 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv04 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv05 = ConvBlock(128, 128, stride=conv_stride, padding=0, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(128, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(256, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        # self.conv5 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool5 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        # self.conv6 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool6 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        # self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(512, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)
        # e = self.conv03(e)
        # e = self.conv04(e)
        e = self.conv05(e)

        e = self.conv1(e)
        e = self.pool1(e)
        e = self.conv2(e)
        e = self.pool2(e)
        e = self.pool_m1(e)

        e = self.conv3(e)
        e = self.pool3(e)
        e = self.conv4(e)
        e = self.pool4(e)
        e = self.pool_m2(e)

        # e = self.conv5(e)
        # e = self.pool5(e)
        # e = self.conv6(e)
        # e = self.pool6(e)
        # e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet3(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = False

        self.conv11 = ConvBlock(3, 64, padding=1, ks=3, has_bn=self.has_bn)
        self.conv12 = ConvBlock(64, 64, padding=1, ks=3, has_bn=self.has_bn)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)  # 16

        self.conv21 = ConvBlock(64, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.conv22 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)  # 8

        self.conv31 = ConvBlock(128, 256, padding=1, ks=3, has_bn=self.has_bn)
        self.conv32 = ConvBlock(256, 256, padding=1, ks=3, has_bn=self.has_bn)
        self.conv33 = ConvBlock(256, 256, padding=1, ks=3, has_bn=self.has_bn)
        self.pool3 = nn.MaxPool2d(2, 2, padding=0)  # 4

        self.conv41 = ConvBlock(256, 512, padding=1, ks=3, has_bn=self.has_bn)
        self.conv42 = ConvBlock(512, 512, padding=1, ks=3, has_bn=self.has_bn)
        self.conv43 = ConvBlock(512, 512, padding=1, ks=3, has_bn=self.has_bn)
        self.pool4 = nn.MaxPool2d(2, 2, padding=0)  # 2

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(512, 10)
        pass

    def forward(self, x):
        e = self.conv11(x)
        e = self.conv12(e)
        e = self.pool1(e)

        e = self.conv21(e)
        e = self.conv22(e)
        e = self.pool2(e)

        e = self.conv31(e)
        e = self.conv32(e)
        e = self.conv33(e)
        e = self.pool3(e)

        e = self.conv41(e)
        e = self.conv42(e)
        e = self.conv43(e)
        e = self.pool4(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet4(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)

        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        # e = self.pool2(e)

        e = self.sp(e)

        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        e = self.pool2(e)

        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e = self.gcn2_conv4(e)
        e = self.pool(e)
        # e = self.pool2(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet5(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(3, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv1 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv2 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn3_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn3_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv1(x)
        e = self.conv2(e)

        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        # e = self.pool2(e)

        e = self.sp(e)

        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        # e = self.pool2(e)

        e = self.sp(e)

        e = self.gcn3_conv1(e)
        e = self.pool(e)
        e = self.gcn3_conv2(e)
        e = self.pool(e)
        # e = self.pool2(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.conv0 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)

        e = self.conv0(e)

        e1 = e
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv2(e)

        e1 = e
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv4(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet7(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)

        e1 = e
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e = self.sp(e)

        e1 = e
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e1 = e
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv4(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet8(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv5 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv6 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)

        e1 = e
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e = self.sp(e)

        e1 = e
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e1 = e
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv4(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e1 = e
        e = self.gcn2_conv5(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv6(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class RunnerSPE(object):

    def __init__(self, model, data_root_path='/mnt/4T/Data/cifar/cifar-10',
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True, image_size=32)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False, image_size=32)

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)

        self.model = model().to(self.device)
        # self.lr_s = [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
        # self.lr_s = [[0, 0.1], [40, 0.01], [70, 0.001], [90, 0.0001]]
        self.lr_s = [[0, 0.1], [100, 0.01], [180, 0.001], [250, 0.0001]]
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][0], momentum=0.9, weight_decay=5e-4)

        self.loss_class = nn.CrossEntropyLoss().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        pass

    def _lr(self, epoch):
        # [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
        pass

    def load_model(self, model_file_name):
        self.model.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs):
        for epoch in range(0, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            epoch_loss, epoch_train_acc = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc = self.test()

            Tools.print('Epoch: {:02d}, lr={:.4f}, Train: {:.4f}/{:.4f} Test: {:.4f}/{:.4f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'],
                epoch_train_acc, epoch_loss, epoch_test_acc, epoch_test_loss))
            pass
        pass

    def _train_epoch(self, print_freq=100):
        self.model.train()
        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for i, (batch_imgs, batch_labels) in enumerate(self.train_loader):
            batch_images = batch_imgs.float().to(self.device)
            batch_labels = batch_labels.long().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model.forward(batch_images)
            loss = self._loss_total(logits, batch_labels)
            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self._accuracy(logits, batch_labels)

            if i % print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_train_acc/nb_data))
                pass
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self, print_freq=50):
        self.model.eval()

        Tools.print()
        epoch_test_acc, nb_data, epoch_test_loss = 0, 0, 0
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.test_loader):
                batch_images = batch_imgs.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                logits = self.model.forward(batch_images)
                loss = self._loss_total(logits, batch_labels)

                nb_data += batch_labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self._accuracy(logits, batch_labels)

                if i % print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1), loss.detach().item(), epoch_test_acc/nb_data))
                    pass
                pass
            pass

        epoch_test_loss /= (len(self.test_loader) + 1)
        epoch_test_acc /= nb_data
        return epoch_test_loss, epoch_test_acc

    def _loss_total(self, logits, batch_labels):
        loss_class = self.loss_class(logits, batch_labels)
        return loss_class

    @staticmethod
    def _save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    @staticmethod
    def _accuracy(scores, targets):
        return (scores.detach().argmax(dim=1) == targets).float().sum().item()

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    """
    # 强数据增强+LR。不确定以下两个哪个带Sigmoid
    GCN  No Sigmoid 2020-04-07 02:50:57 Epoch: 75, lr=0.0000, Train: 0.5148/1.4100 Test: 0.5559/1.3145
    GCN Has Sigmoid 2020-04-07 07:35:40 Epoch: 72, lr=0.0000, Train: 0.5354/1.3428 Test: 0.5759/1.2394
    GCN  No Sigmoid 2020-04-08 06:36:51 Epoch: 70, lr=0.0000, Train: 0.5099/1.4281 Test: 0.5505/1.3224
    GCN Has Sigmoid 2020-04-08 07:24:54 Epoch: 73, lr=0.0001, Train: 0.5471/1.3164 Test: 0.5874/1.2138

    # 原始:数据增强+LR
    GCN           No Sigmoid 2020-04-08 06:24:55 Epoch: 98, lr=0.0001, Train: 0.6696/0.9954 Test: 0.6563/1.0695
    GCN          Has Sigmoid 2020-04-08 15:41:33 Epoch: 97, lr=0.0001, Train: 0.7781/0.6535 Test: 0.7399/0.8137
    GraphSageNet Has Sigmoid 2020-04-08 23:31:25 Epoch: 88, lr=0.0001, Train: 0.8074/0.5703 Test: 0.7612/0.7322
    GatedGCNNet  Has Sigmoid 2020-04-10 03:55:12 Epoch: 92, lr=0.0001, Train: 0.8401/0.4779 Test: 0.7889/0.6741

    CNNNet 2x2 3x3 4layer       2020-04-13 15:18:12 Epoch: 96, lr=0.0001, Train: 0.8049/0.5423 Test: 0.7505/0.7542
    CNNNet 2x2 3x3 4layer       2020-04-13 15:18:12 Epoch: 96, lr=0.0001, Train: 0.8049/0.5423 Test: 0.7505/0.7542
    CNNNet 4x4 3x3 4layer       2020-04-13 13:35:02 Epoch: 99, lr=0.0001, Train: 0.7597/0.6751 Test: 0.7239/0.8237
    CNNNet 6x6 3x3 4layer       2020-04-13 16:18:59 Epoch: 98, lr=0.0001, Train: 0.6807/0.8896 Test: 0.6581/0.9817
    
    CNNNet 4x4 3x3 5layer       2020-04-13 14:28:55 Epoch: 98, lr=0.0001, Train: 0.7594/0.6753 Test: 0.7224/0.8188
    CNNNet 4x4 3x3 4layer       2020-04-13 13:35:02 Epoch: 99, lr=0.0001, Train: 0.7597/0.6751 Test: 0.7239/0.8237
    CNNNet 4x4 3x3 3layer       2020-04-13 14:33:04 Epoch: 95, lr=0.0001, Train: 0.7515/0.6934 Test: 0.7223/0.8088
    
    CNNNet 4x4 3x3 4layer + add 2020-04-13 14:11:01 Epoch: 99, lr=0.0001, Train: 0.7632/0.6645 Test: 0.7285/0.8094
    
    CNNNet 2x2 3x3 5layer       2020-04-13 15:12:25 Epoch: 85, lr=0.0001, Train: 0.8131/0.5320 Test: 0.7515/0.7446
    CNNNet 1x1 3x3 6layer       2020-04-13 16:51:52 Epoch: 97, lr=0.0001, Train: 0.6310/1.0423 Test: 0.5493/1.3548
    CNNNet 2x2 3x3 6layer       2020-04-13 15:29:08 Epoch: 84, lr=0.0001, Train: 0.8114/0.5296 Test: 0.7536/0.7498
    
    CNNNet 4x4 5x5 4layer       2020-04-13 16:22:10 Epoch: 98, lr=0.0001, Train: 0.7357/0.7381 Test: 0.6981/0.8875
    
    CNNNet 2x2 3x3 4layer 2pool 2020-04-13 17:09:29 Epoch: 90, lr=0.0001, Train: 0.8121/0.5278 Test: 0.7654/0.7248
    CNNNet 4x4 3x3 4layer 2pool 2020-04-13 16:49:49 Epoch: 97, lr=0.0001, Train: 0.7588/0.6757 Test: 0.7210/0.8182
    CNNNet 1x1 3x3 4layer 2pool 2020-04-13 19:08:47 Epoch: 77, lr=0.0001, Train: 0.6196/1.0698 Test: 0.5481/1.3280
    CNNNet 2x2 3x3 4layer 2pool 2020-04-13 18:47:49 Epoch: 97, lr=0.0001, Train: 0.8093/0.5348 Test: 0.7624/0.7335
    CNNNet 4x4 3x3 4layer 2pool 2020-04-13 18:39:08 Epoch: 85, lr=0.0001, Train: 0.7521/0.6975 Test: 0.7242/0.8246
    CNNNet 1x1 3x3 6layer 3pool 2020-04-13 19:27:36 Epoch: 97, lr=0.0001, Train: 0.6668/0.9343 Test: 0.5684/1.3213
    CNNNet 2x2 3x3 6layer 3pool 2020-04-13 18:40:26 Epoch: 96, lr=0.0001, Train: 0.8274/0.4786 Test: 0.7625/0.7672
    CNNNet 4x4 3x3 6layer 3pool 2020-04-13 18:36:38 Epoch: 96, lr=0.0001, Train: 0.7562/0.6835 Test: 0.7123/0.8373
    
    CNNNet 1x1 3x3 4layer  3x3  2020-04-13 21:12:20 Epoch: 98, lr=0.0001, Train: 0.9783/0.0610 Test: 0.8743/0.6718
    CNNNet 1x1 3x3 6layer  3x3  2020-04-13 21:11:06 Epoch: 99, lr=0.0001, Train: 0.9713/0.0822 Test: 0.8507/0.7306
    CNNNet 2x2 3x3 4layer  3x3  2020-04-13 19:44:08 Epoch: 96, lr=0.0001, Train: 0.9515/0.1366 Test: 0.8501/0.6302
    CNNNet 2x2 3x3 6layer  3x3  2020-04-13 19:40:44 Epoch: 82, lr=0.0001, Train: 0.9227/0.2124 Test: 0.8353/0.6319
    CNNNet 4x4 3x3 4layer  3x3  2020-04-13 20:44:20 Epoch: 94, lr=0.0001, Train: 0.8873/0.3205 Test: 0.8177/0.6267
    CNNNet 4x4 3x3 6layer  3x3  2020-04-13 21:28:28 Epoch: 97, lr=0.0001, Train: 0.8710/0.3698 Test: 0.8059/0.6274
    
    CNNNet 4x4 3x3 4layer   4   2020-04-13 13:35:02 Epoch: 99, lr=0.0001, Train: 0.7597/0.6751 Test: 0.7239/0.8237
    CNNNet 4x4 3x3 4layer  2,4  2020-04-14 13:04:28 Epoch: 97, lr=0.0001, Train: 0.7817/0.6075 Test: 0.7349/0.8027
    CNNNet 4x4 3x3 4layer 12,34 2020-04-14 13:11:27 Epoch: 96, lr=0.0001, Train: 0.7869/0.5908 Test: 0.7287/0.8226
    """
    """
    CNNNet 4x4 3x3 4layer 1111 pool 2020-04-14 16:03:04 Epoch: 99, lr=0.0001, Train: 0.7604/0.6727 Test: 0.7221/0.8298
    CNNNet 4x4 3x3 4layer 3111 pool 2020-04-14 16:30:18 Epoch: 87, lr=0.0001, Train: 0.8586/0.4000 Test: 0.7951/0.6443
    CNNNet 4x4 3x3 4layer 3311 pool 2020-04-14 16:35:09 Epoch: 95, lr=0.0001, Train: 0.8744/0.3517 Test: 0.8155/0.6207
    CNNNet 4x4 3x3 4layer 1133 pool 2020-04-14 17:11:43 Epoch: 96, lr=0.0001, Train: 0.8203/0.5004 Test: 0.7798/0.6776
    CNNNet 4x4 3x3 4layer 3333 pool 2020-04-14 16:46:42 Epoch: 99, lr=0.0001, Train: 0.8999/0.2810 Test: 0.8268/0.6054
    
    CNNNet 2x2 3x3 4layer 1111 pool 2020-04-14 20:12:28 Epoch: 92, lr=0.0001, Train: 0.9064/0.2621 Test: 0.8252/0.6181
    CNNNet 2x2 3x3 6layer 1111 pool 2020-04-14 21:05:32 Epoch: 91, lr=0.0001, Train: 0.8955/0.2918 Test: 0.8123/0.6613
    """
    """
    # + Conv
    CNNNet 4x4 3x3 4layer 1111 pool 2020-04-14 18:09:44 Epoch: 97, lr=0.0001, Train: 0.8851/0.3217 Test: 0.8185/0.6179
    CNNNet 4x4 3x3 6layer 1111 pool 2020-04-14 19:26:49 Epoch: 99, lr=0.0001, Train: 0.8823/0.3221 Test: 0.8095/0.6584
    CNNNet 4x4 3x3 6layer 1111 pool 2020-04-14 21:03:49 Epoch: 93, lr=0.0001, Train: 0.8558/0.4000 Test: 0.7925/0.6916
    CNNNet 4x4 5x5 4layer 1111 pool 2020-04-14 19:30:13 Epoch: 94, lr=0.0001, Train: 0.8643/0.3786 Test: 0.8011/0.6882
    CNNNet 4x4 5x5 6layer 1111 pool 2020-04-14 19:38:09 Epoch: 99, lr=0.0001, Train: 0.8648/0.3780 Test: 0.8028/0.6647
    
    CNNNet 4x4 3x3 4layer 1111 pool 2020-04-14 18:09:44 Epoch: 97, lr=0.0001, Train: 0.8851/0.3217 Test: 0.8185/0.6179
    CNNNet 4x4 3x3 4layer 1111 pool moreconv 2020-04-14 23:34:24 Epoch: 96, lr=0.0001, Train: 0.9130/0.2456 Test: 0.8292/0.6251
    """
    """
    BN 2x2 4layer 1111 pool 3conv 2020-04-15 00:50:46 Epoch: 72, lr=0.0001, Train: 0.9391/0.1809 Test: 0.8712/0.4178
    BN 4x4 4layer 1111 pool 3conv 2020-04-15 00:34:39 Epoch: 97, lr=0.0001, Train: 0.9166/0.2327 Test: 0.8572/0.4632
    
    BN 2x2 4layer 1111 pool 3conv large 2020-04-15 01:44:01 Epoch: 88, lr=0.0001, Train: 0.9876/0.0460 Test: 0.8937/0.4051
    BN 4x4 4layer 1111 pool 3conv large 2020-04-15 01:37:42 Epoch: 94, lr=0.0001, Train: 0.9776/0.0686 Test: 0.8821/0.4486
    """
    """
    CNNNet4 1919626               2020-04-26 16:39:48 Epoch: 200, lr=0.0010, Train: 0.9989/0.0071 Test: 0.9205/0.3287
    CNNNet4  171293 1sp 0pool     2020-04-26 18:48:18 Epoch: 188, lr=0.0010, Train: 0.9928/0.0285 Test: 0.8868/0.4595
    CNNNet4  171293 1sp 1pool     2020-04-26 21:39:03 Epoch: 209, lr=0.0010, Train: 0.9965/0.0155 Test: 0.8904/0.5183
    CNNNet5  171293 2sp 0pool     2020-04-27 00:15:34 Epoch: 182, lr=0.0010, Train: 0.9896/0.0400 Test: 0.8939/0.4049
    CNNNet4  171293 1sp 1pool da  2020-04-27 00:55:13 Epoch: 196, lr=0.0010, Train: 0.9634/0.1086 Test: 0.8791/0.4591
    CNNNet6  258309 1sp 0pool Res 2020-04-27 17:35:29 Epoch: 230, lr=0.0010, Train: 0.9990/0.0070 Test: 0.9034/0.4176
    CNNNet7  230743 1sp 0pool Res 2020-04-28 19:52:45 Epoch: 187, lr=0.0010, Train: 0.9914/0.0351 Test: 0.8905/0.4287
    CNNNet8  274251 1sp 0pool Res 
    """

    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("CNNNet")
    # _num_workers = 2
    # _use_gpu = False
    # _gpu_id = "1"

    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    _root_ckpt_dir = "./ckpt2/dgl/Test_1/{}".format("CNNNet")
    _num_workers = 4
    _use_gpu = True
    _gpu_id = "0"

    Tools.print("ckpt:{}, workers:{}, gpu:{}".format(_root_ckpt_dir, _num_workers, _gpu_id))

    runner = RunnerSPE(model=CNNNet8, data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(300)

    pass
