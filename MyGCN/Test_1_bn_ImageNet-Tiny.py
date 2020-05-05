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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=64):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path

        # self.tran_train = transforms.Compose([transforms.RandomResizedCrop(image_size),
        #                                       transforms.RandomHorizontalFlip(), transforms.ToTensor()])

        self.tran_train = transforms.Compose([transforms.RandomCrop(image_size, padding=8),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor()])

        _dir = os.path.join(self.data_root_path, "train" if is_train else "val_new")

        self.tran_test = transforms.Compose([transforms.ToTensor()])

        self.data_set = datasets.ImageFolder(root=_dir,
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
        self.has_bn = True
        conv_stride = 6
        avg_range = 3

        self.conv0 = ConvBlock(3, 64, stride=conv_stride, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(128, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv5 = ConvBlock(256, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.pool5 = nn.AvgPool2d(3, 1, padding=1)
        self.conv6 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.pool6 = nn.AvgPool2d(3, 1, padding=1)
        self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(512, 200)
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

        e = self.conv5(e)
        e = self.pool5(e)
        e = self.conv6(e)
        e = self.pool6(e)
        e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.conv0 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.conv1 = ConvBlock(128, 256, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.conv2 = ConvBlock(256, 512, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.conv3 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(512, 10)
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


class CNNNet7_10(nn.Module):

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


class CNNNet6_11(nn.Module):

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


class CNNNet7_12(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=False)

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
        e = F.relu(e)
        e = self.gcn1_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = F.relu(e)

        e = self.sp(e)

        e1 = e
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e2 = e
        e = F.relu(e)
        e = self.gcn2_conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = F.relu(e)

        e1 = e
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e2 = e
        e = F.relu(e)
        e = self.gcn2_conv4(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = F.relu(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class ResBlock(nn.Module):

    def __init__(self, cin=146, cout=146, padding=0, ks=1, has_relu=True, has_bn=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.pool = nn.AvgPool2d(3, 1, padding=1)

        self.conv1 = ConvBlock(cin, cout, padding=padding, ks=ks, has_bn=self.has_bn, has_relu=self.has_relu)
        self.conv2 = ConvBlock(cout, cout, padding=padding, ks=ks, has_bn=self.has_bn, has_relu=self.has_relu)
        pass

    def forward(self, e):
        e1 = e
        e = self.conv1(e)
        e = self.pool(e)
        e2 = e
        if not self.has_relu:
            e = F.relu(e)
        e = self.conv2(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        if not self.has_relu:
            e = F.relu(e)
        return e

    pass


class CNNNet7_13(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = True
        self.has_relu = True
        sp_size = 6
        sp_size_padding = 2
        # sp_size = 4
        # sp_size_padding = 0

        self.sp = nn.AvgPool2d(sp_size, sp_size, padding=sp_size_padding)

        self.conv1 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv2 = ConvBlock(64, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_1 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn1_2 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn1_3 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)

        self.gcn2_1 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn2_2 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn2_3 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn2_4 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn2_5 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)
        self.gcn2_6 = ResBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn, has_relu=self.has_relu)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv1(x)
        e = self.conv2(e)

        e = self.gcn1_1(e)
        e = self.gcn1_2(e)
        e = self.gcn1_3(e)

        # e = self.sp(e)

        e = self.gcn2_1(e)
        e = self.gcn2_2(e)
        e = self.gcn2_3(e)
        e = self.gcn2_4(e)
        e = self.gcn2_5(e)
        e = self.gcn2_6(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_14(nn.Module):

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
        e = self.pool(e)
        e = self.gcn1_conv1(e)
        e2 = e
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv1(e)
        e2 = e
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv2(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv3(e)
        e2 = e
        e = self.pool(e)
        e = self.gcn2_conv4(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_15(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(3, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv11 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv12 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv0 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv21 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv21 = ConvBlock(64, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv22 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

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
        e = self.conv11(x)
        e = self.conv12(e)

        e = self.pool2(e)
        e = self.conv0(e)

        e = self.conv21(e)
        e = self.conv22(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv2(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e = self.gcn2_conv4(e)
        e3 = e
        e = e1 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(2, 2, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv11 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv12 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv0 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv21 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv22 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv31 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv32 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

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
        e = self.conv11(x)
        e = self.conv12(e)
        e = self.pool2(e)

        e = self.conv0(e)

        e = self.conv21(e)
        e = self.conv22(e)
        e = self.pool2(e)

        e = self.conv31(e)
        e = self.conv32(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv2(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e = self.gcn2_conv4(e)
        e3 = e
        e = e1 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_17(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(2, 2, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv11 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv12 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        # self.conv0 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.conv21 = ConvBlock(64, 128, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv22 = ConvBlock(128, 128, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv31 = ConvBlock(128, 256, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv32 = ConvBlock(256, 256, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(256, 256, padding=0, ks=1, has_bn=self.has_bn)
        self.conv1 = ConvBlock(256, 512, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.conv2 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)
        self.conv3 = ConvBlock(512, 512, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(512, 10)
        pass

    def forward(self, x):
        e = self.conv11(x)
        e = self.conv12(e)
        e = self.pool2(e)

        # e = self.conv0(e)

        e = self.conv21(e)
        e = self.conv22(e)
        e = self.pool2(e)

        e = self.conv31(e)
        e = self.conv32(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn1_conv1(e)
        e = self.pool(e)
        e = self.gcn1_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv1(e)
        e = self.pool(e)
        e = self.gcn2_conv2(e)
        e3 = e
        e = e1 + e3
        e = self.conv2(e)

        e1 = e
        e = self.pool(e)
        e = self.gcn2_conv3(e)
        e = self.pool(e)
        e = self.gcn2_conv4(e)
        e3 = e
        e = e1 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_18(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(4, 4, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv01 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv02 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.conv0 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn1_conv11 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv12 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv21 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn1_conv22 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv11 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv12 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv21 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv22 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv31 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv32 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv41 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.gcn2_conv42 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv01(x)
        e = self.conv02(e)

        e = self.conv0(e)

        e1 = e
        e = self.gcn1_conv11(e)
        e = self.gcn1_conv12(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn1_conv21(e)
        e = self.gcn1_conv22(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        e = self.gcn2_conv11(e)
        e = self.gcn2_conv12(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv21(e)
        e = self.gcn2_conv22(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv2(e)

        e1 = e
        e = self.gcn2_conv31(e)
        e = self.gcn2_conv32(e)
        e = self.pool(e)
        e2 = e
        e = self.gcn2_conv41(e)
        e = self.gcn2_conv42(e)
        e = self.pool(e)
        e3 = e
        e = e1 + e2 + e3
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_16_19(nn.Module):

    def __init__(self):
        super().__init__()
        self.has_bn = True

        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.sp = nn.AvgPool2d(2, 2, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv11 = ConvBlock(3, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv12 = ConvBlock(64, 64, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv0 = ConvBlock(64, 146, padding=0, ks=1, has_bn=self.has_bn)
        self.conv21 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv22 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv31 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)
        self.conv32 = ConvBlock(146, 146, stride=1, padding=1, ks=3, has_bn=self.has_bn)

        self.gcn1_conv1 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.gcn1_conv2 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.conv1 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv1 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.gcn2_conv2 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.conv2 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.gcn2_conv3 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.gcn2_conv4 = ConvBlock(146, 146, padding=1, ks=3, has_bn=self.has_bn)
        self.conv3 = ConvBlock(146, 146, padding=0, ks=1, has_bn=self.has_bn)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(146, 10)
        pass

    def forward(self, x):
        e = self.conv11(x)
        e = self.conv12(e)
        e = self.pool2(e)

        e = self.conv0(e)

        e = self.conv21(e)
        e = self.conv22(e)
        e = self.pool2(e)

        e = self.conv31(e)
        e = self.conv32(e)

        e1 = e
        # e = self.pool(e)
        e = self.gcn1_conv1(e)
        # e = self.pool(e)
        e = self.gcn1_conv2(e)
        e = e1 + e
        e = self.conv1(e)

        e = self.sp(e)

        e1 = e
        # e = self.pool(e)
        e = self.gcn2_conv1(e)
        # e = self.pool(e)
        e = self.gcn2_conv2(e)
        e = e1 + e
        e = self.conv2(e)

        e1 = e
        # e = self.pool(e)
        e = self.gcn2_conv3(e)
        # e = self.pool(e)
        e = self.gcn2_conv4(e)
        e = e1 + e
        e = self.conv3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet6_20(nn.Module):

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

        # e = self.pool2(e)

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


class RunnerSPE(object):

    def __init__(self, model, data_root_path='/mnt/4T/Data/cifar/cifar-10', weight_decay=5e-4, is_sgd=False,
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True, image_size=64)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False, image_size=64)

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)

        self.model = model().to(self.device)

        if not is_sgd:
            self.lr_s = [[0, 0.01], [33, 0.001], [66, 0.0001]]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)
        else:
            self.lr_s = [[0, 0.1], [100, 0.01], [180, 0.001], [250, 0.0001]]
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][0],
                                             momentum=0.9, weight_decay=weight_decay)
            pass

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
    CNNNet 2020-05-05 11:55:06 Epoch: 53, lr=0.0002, Train: 0.4655/2.1123 Test: 0.3340/3.0345
    CNNNet 2020-05-05 13:53:30 Epoch: 80, lr=0.0000, Train: 0.5292/1.8123 Test: 0.3303/3.2726
    """

    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("CNNNet")
    # _num_workers = 2
    # _use_gpu = False
    # _gpu_id = "1"

    # _data_root_path = '/mnt/4T/Data/tiny-imagenet-200/tiny-imagenet-200'
    _data_root_path = '/home/ubuntu/ALISURE/data/tiny-imagenet-200'
    _root_ckpt_dir = "./ckpt2/dgl/Test_1_tiny-imagenet/{}".format("CNNNet")
    _weight_decay = 5e-4
    _num_workers = 4
    _is_sgd = False
    _use_gpu = True
    # _gpu_id = "0"
    _gpu_id = "1"

    Tools.print("ckpt:{}, workers:{}, gpu:{}".format(_root_ckpt_dir, _num_workers, _gpu_id))

    runner = RunnerSPE(model=CNNNet, data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, is_sgd=_is_sgd,
                       weight_decay=_weight_decay, num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(100)

    pass
