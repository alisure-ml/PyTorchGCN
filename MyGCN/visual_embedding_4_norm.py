"""
    视觉嵌入：图像 =》超像素 =》嵌入 =》重构
    输出为图卷积的输入
"""
import os
import cv2
import time
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from skimage import io
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DealSuperPixel(object):

    def __init__(self, image_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_size = super_pixel_size
        self.super_pixel_num = (self.ds_image_size // self.super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        self.slic_sigma = slic_sigma
        self.slic_max_iter = slic_max_iter
        pass

    def run(self):
        segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                    sigma=self.slic_sigma, max_iter=self.slic_max_iter)

        super_pixel_info = {}
        for i in range(segment.max() + 1):
            _now_i = segment == i
            _now_where = np.argwhere(_now_i)
            x_min, x_max = _now_where[:, 0].min(), _now_where[:, 0].max()
            y_min, y_max = _now_where[:, 1].min(), _now_where[:, 1].max()

            # 大小
            sp_i_size = len(_now_where) / (self.super_pixel_size * self.super_pixel_size) - 1
            # 坐标
            super_pixel_area = (x_min, x_max, y_min, y_max)
            # 是否属于超像素
            super_pixel_label = np.asarray(_now_i[x_min: x_max + 1, y_min: y_max + 1], dtype=np.int)
            super_pixel_label = np.expand_dims(super_pixel_label, axis=-1)
            # 属于超像素所在矩形区域的值
            super_pixel_data = self.image_data[x_min: x_max + 1, y_min: y_max + 1]
            # 属于超像素的值, 不属于超像素的值设置为-255，用来区分不属于超像素的点和[0,0,0]
            _super_pixel_label_3 = np.concatenate([super_pixel_label, super_pixel_label, super_pixel_label], axis=-1)
            super_pixel_data2 = np.array(super_pixel_data, dtype=np.int)
            super_pixel_data2[_super_pixel_label_3==0] = -255

            # 计算邻接矩阵
            _x_min_a = x_min - (1 if x_min > 0 else 0)
            _y_min_a = y_min - (1 if y_min > 0 else 0)
            _x_max_a = x_max + 1 + (1 if x_max < len(segment) else 0)
            _y_max_a = y_max + 1 + (1 if y_max < len(segment[0]) else 0)
            super_pixel_area_large = segment[_x_min_a: _x_max_a, _y_min_a: _y_max_a]
            super_pixel_unique_id = np.unique(super_pixel_area_large)
            super_pixel_adjacency = [sp_id for sp_id in super_pixel_unique_id if sp_id != i]

            # 结果
            super_pixel_info[i] = {"size": sp_i_size, "area": super_pixel_area,
                                   "label": super_pixel_label, "data": super_pixel_data,
                                   "data2": super_pixel_data2, "adj": super_pixel_adjacency}
            pass

        adjacency_info = []
        for super_pixel_id in super_pixel_info:
            now_adj = super_pixel_info[super_pixel_id]["adj"]
            now_area = super_pixel_info[super_pixel_id]["area"]

            _adjacency_area = [super_pixel_info[sp_id]["area"] for sp_id in now_adj]
            _now_center = ((now_area[0] + now_area[1]) / 2, (now_area[2] + now_area[3]) / 2)
            _adjacency_center = [((area[0] + area[1]) / 2, (area[2] + area[3]) / 2) for area in _adjacency_area]

            adjacency_dis = [np.sqrt((_now_center[0] - center[0]) ** 2 +
                                     (_now_center[1] - center[1]) ** 2) for center in _adjacency_center]
            softmax_w = self._softmax_of_distance(adjacency_dis)
            adjacency_w = [(super_pixel_id, adj_id, softmax_w[ind]) for ind, adj_id in enumerate(now_adj)]

            adjacency_info.extend(adjacency_w)
            pass

        return segment, super_pixel_info, adjacency_info

    def show(self, segment):
        result = segmentation.mark_boundaries(self.image_data, segment)
        fig = plt.figure("{}".format(self.super_pixel_num))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(result)
        plt.axis("off")
        plt.show()
        pass

    @staticmethod
    def _softmax_of_distance(distance):
        distance = np.asarray(distance) + 1
        distance = np.sum(distance) / distance
        exp_distance= np.exp(distance)
        return exp_distance / np.sum(exp_distance, axis=0)

    @staticmethod
    def demo():
        now_image_name = "data\\input\\1.jpg"
        now_image_data = io.imread(now_image_name)
        deal_super_pixel = DealSuperPixel(image_data=now_image_data, ds_image_size=224)
        now_segment, now_super_pixel_info, now_adjacency_info = deal_super_pixel.run()
        deal_super_pixel.show(now_segment)
        pass

    pass


class MyCIFAR10(datasets.CIFAR10):

    # def __init__(self, root, train, transform):
    #     super().__init__(root=root, train=train, transform=transform)
    #     self.data = self.data[:10]
    #     self.targets = self.targets[:10]
    #     pass

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return np.asarray(img), target

    pass


class DataCIFAR10(object):

    def __init__(self, train_batch=32, test_batch=32, data_root_path='/mnt/4T/Data/cifar/cifar-10'):
        # Data
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        # transform_test = transforms.Compose([transforms.ToTensor()])

        train_set = MyCIFAR10(root=data_root_path, train=True, download=True, transform=transform_train)
        self.train_loader = data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=2)

        test_set = MyCIFAR10(root=data_root_path, train=False, download=False, transform=None)
        self.test_loader = data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=2)
        pass

    pass


class Normalize(nn.Module):

    def __init__(self, power=2):
        super().__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

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


class ConvTransposeBlock(nn.Module):

    def __init__(self, cin, cout, stride=2, padding=1, has_relu=True):
        super().__init__()
        self.has_relu = has_relu

        self.conv = nn.ConvTranspose2d(cin, cout, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    pass


class EmbeddingNet(nn.Module):

    def __init__(self):
        super().__init__()
        # input 24
        self.conv11 = ConvBlock(3, 64, 1, padding=1, has_relu=True)  # 24
        self.conv12 = ConvBlock(64, 64, 1, padding=1, has_relu=True)  # 24
        self.pool1 = nn.MaxPool2d(2, 2)  # 12
        self.conv21 = ConvBlock(64, 128, 1, padding=1, has_relu=True)  # 12
        self.conv22 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 12
        self.pool2 = nn.MaxPool2d(2, 2)  # 6
        self.conv31 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 6
        self.conv32 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 6
        self.pool3 = nn.MaxPool2d(2, 2)  # 3

        self.conv_shape = ConvBlock(128, 32, 1, padding=0, has_relu=True, bias=False)  # 1
        self.conv_texture = ConvBlock(128, 32, 1, padding=0, has_relu=True, bias=False)  # 1

        self.shape_conv0 = ConvTransposeBlock(32, 128, padding=0, has_relu=True)  # 3
        self.shape_conv1 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 3
        self.shape_up1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.shape_conv2 = ConvBlock(128, 64, padding=1, has_relu=True)  # 6
        self.shape_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 12
        self.shape_conv3 = ConvBlock(64, 64, padding=1, has_relu=True)  # 12
        self.shape_up3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 24
        self.shape_out = ConvBlock(64, 1, padding=1, has_relu=False)  # 24

        self.texture_conv0 = ConvTransposeBlock(64, 128, padding=0, has_relu=True)  # 3
        self.texture_conv1 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 3
        self.texture_up1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.texture_conv2 = ConvBlock(128, 64, padding=1, has_relu=True)  # 6
        self.texture_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 12
        self.texture_conv3 = ConvBlock(64, 64, padding=1, has_relu=True)  # 12
        self.texture_up3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 24
        self.texture_out = ConvBlock(64, 3, padding=1, has_relu=False)  # 24
        pass

    def forward(self, x):
        if x.size()[2] != 24 or x.size()[3] != 24:
            x = torch.nn.functional.interpolate(x, size=[24, 24], mode='bilinear')
            pass

        e1 = self.conv12(self.conv11(x))
        p1 = self.pool1(e1)
        e2 = self.conv22(self.conv21(p1))
        p2 = self.pool2(e2)
        e3 = self.conv32(self.conv31(p2))
        p3 = self.pool3(e3)
        shape = self.conv_shape(p3)
        texture = self.conv_texture(p3)

        shape_d0 = self.shape_conv0(shape)
        shape_d1 = self.shape_up1(self.shape_conv1(shape_d0))
        shape_d2 = self.shape_up2(self.shape_conv2(shape_d1))
        shape_d3 = self.shape_up3(self.shape_conv3(shape_d2))
        shape_out = self.shape_out(shape_d3)

        texture_d0 = self.texture_conv0(torch.cat([texture, shape], dim=1))
        texture_d1 = self.texture_up1(self.texture_conv1(texture_d0))
        texture_d2 = self.texture_up2(self.texture_conv2(texture_d1))
        texture_d3 = self.texture_up3(self.texture_conv3(texture_d2))
        texture_out = self.texture_out(texture_d3)

        shape_feature = shape.view(shape.size()[0], -1)
        texture_feature = texture.view(texture.size()[0], -1)

        return shape_feature, texture_feature, shape_out, texture_out

    pass


class EmbeddingNetCIFAR(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_size = 6

        # input 24
        self.conv11 = ConvBlock(3, 64, 1, padding=1, has_relu=True)  # 6
        self.conv12 = ConvBlock(64, 128, 1, padding=1, has_relu=True)  # 6
        self.pool1 = nn.MaxPool2d(2, 2)  # 3
        self.conv21 = ConvBlock(128, 128, 1, padding=1, has_relu=True)  # 3

        self.conv_shape = ConvBlock(128, 32, 1, padding=0, has_relu=True, bias=False)  # 1
        self.conv_texture = ConvBlock(128, 32, 1, padding=0, has_relu=True, bias=False)  # 1

        self.shape_conv0 = ConvTransposeBlock(32, 128, padding=0, has_relu=True)  # 3
        self.shape_conv1 = ConvBlock(128, 64, 1, padding=1, has_relu=True)  # 3
        self.shape_up1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.shape_conv2 = ConvBlock(64, 64, padding=1, has_relu=True)  # 6
        self.shape_out = ConvBlock(64, 1, padding=1, has_relu=False)  # 6

        self.texture_conv0 = ConvTransposeBlock(64, 128, padding=0, has_relu=True)  # 3
        self.texture_conv1 = ConvBlock(128, 64, 1, padding=1, has_relu=True)  # 3
        self.texture_up1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.texture_conv2 = ConvBlock(64, 64, padding=1, has_relu=True)  # 6
        self.texture_out = ConvBlock(64, 3, padding=1, has_relu=False)  # 6
        pass

    def forward(self, x):
        if x.size()[2] != self.input_size or x.size()[3] != self.input_size:
            raise Exception("1234567890")

        e1 = self.conv12(self.conv11(x))
        p1 = self.pool1(e1)
        e2 = self.conv21(p1)
        shape = self.conv_shape(e2)
        texture = self.conv_texture(e2)

        shape_d0 = self.shape_conv0(shape)
        shape_d1 = self.shape_up1(self.shape_conv1(shape_d0))
        shape_d2 = self.shape_conv2(shape_d1)
        shape_out = self.shape_out(shape_d2)

        texture_d0 = self.texture_conv0(torch.cat([texture, shape], dim=1))
        texture_d1 = self.texture_up1(self.texture_conv1(texture_d0))
        texture_d2 = self.texture_conv2(texture_d1)
        texture_out = self.texture_out(texture_d2)

        shape_feature = shape.view(shape.size()[0], -1)
        texture_feature = texture.view(texture.size()[0], -1)

        return shape_feature, texture_feature, shape_out, texture_out

    pass


class EmbeddingNetCIFARSmallNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_size = 6

        # input 24
        self.conv1 = ConvBlock(3, 64, 1, padding=1, has_relu=True)  # 6
        self.pool1 = nn.MaxPool2d(2, 2)  # 3
        self.conv2 = ConvBlock(64, 64, 1, padding=1, has_relu=True)  # 3

        self.conv_shape = ConvBlock(64, 16, 1, padding=0, has_relu=True, bias=False)  # 1
        self.conv_texture = ConvBlock(64, 16, 1, padding=0, has_relu=True, bias=False)  # 1

        self.shape_up1 = ConvTransposeBlock(16, 64, padding=0, has_relu=True)  # 3
        self.shape_conv1 = ConvBlock(64, 64, 1, padding=1, has_relu=True)  # 3
        self.shape_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.shape_conv2 = ConvBlock(64, 64, padding=1, has_relu=True)  # 6
        self.shape_out = ConvBlock(64, 1, padding=1, has_relu=False)  # 6

        self.texture_up1 = ConvTransposeBlock(32, 64, padding=0, has_relu=True)  # 3
        self.texture_conv1 = ConvBlock(64, 64, 1, padding=1, has_relu=True)  # 3
        self.texture_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.texture_conv2 = ConvBlock(64, 64, padding=1, has_relu=True)  # 6
        self.texture_out = ConvBlock(64, 3, padding=1, has_relu=False)  # 6

        self.norm = Normalize()
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        if x.size()[2] != self.input_size or x.size()[3] != self.input_size:
            raise Exception("1234567890")

        e1 = self.pool1(self.conv1(x))
        e2 = self.conv2(e1)

        shape = self.conv_shape(e2)
        texture = self.conv_texture(e2)
        shape_norm = self.norm(shape)
        texture_norm = self.norm(texture)

        shape_feature = shape_norm.view(shape_norm.size()[0], -1)
        texture_feature = texture_norm.view(texture_norm.size()[0], -1)

        shape_d0 = self.shape_conv1(self.shape_up1(shape_norm))
        shape_d1 = self.shape_conv2(self.shape_up2(shape_d0))
        shape_out = self.sigmoid(self.shape_out(shape_d1))

        texture_d0 = self.texture_conv1(self.texture_up1(torch.cat([texture_norm, shape_norm], dim=1)))
        texture_d1 = self.texture_conv2(self.texture_up2(texture_d0))
        texture_out = self.sigmoid(self.texture_out(texture_d1))

        return shape_feature, texture_feature, shape_out, texture_out

    pass


class EmbeddingNetCIFARSmallNorm2(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_size = 6
        self.embedding_size = 16
        self.has_bn = False

        # input 24
        self.conv11 = ConvBlock(3, 32, 1, padding=1, has_relu=True, has_bn=self.has_bn)  # 6
        self.conv12 = ConvBlock(32, 32, 1, padding=1, has_relu=True, has_bn=self.has_bn)  # 3
        self.pool1 = nn.MaxPool2d(2, 2)  # 3
        self.conv21 = ConvBlock(32, 64, 1, padding=1, has_relu=True, has_bn=self.has_bn)  # 3
        self.conv22 = ConvBlock(64, 64, 1, padding=1, has_relu=True, has_bn=self.has_bn)  # 3

        self.conv_shape = ConvBlock(64, self.embedding_size, 1, 0, has_relu=True, has_bn=False, bias=self.has_bn)  # 1
        self.conv_texture = ConvBlock(64, self.embedding_size, 1, 0, has_relu=True, has_bn=False, bias=self.has_bn)

        self.shape_up1 = nn.UpsamplingBilinear2d(scale_factor=3)  # 3
        self.shape_conv1 = ConvBlock(self.embedding_size, 64, 1, padding=1, has_relu=True, has_bn=self.has_bn)  # 3
        self.shape_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.shape_conv2 = ConvBlock(64, 32, padding=1, has_relu=True, has_bn=self.has_bn)  # 6
        self.shape_out = ConvBlock(32, 1, padding=1, has_relu=False, has_bn=self.has_bn)  # 6

        self.texture_up1 = nn.UpsamplingBilinear2d(scale_factor=3) # 3
        self.texture_conv1 = ConvBlock(self.embedding_size * 2, 128, 1, 1, has_relu=True, has_bn=self.has_bn)  # 3
        self.texture_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.texture_conv2 = ConvBlock(128, 32, padding=1, has_relu=True, has_bn=self.has_bn)  # 6
        self.texture_out = ConvBlock(32, 3, padding=1, has_relu=False, has_bn=self.has_bn)  # 6

        self.norm = Normalize()
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        if x.size()[2] != self.input_size or x.size()[3] != self.input_size:
            raise Exception("1234567890")

        e1 = self.pool1(self.conv12(self.conv11(x)))
        e2 = self.conv22(self.conv21(e1))

        shape = self.conv_shape(e2)
        texture = self.conv_texture(e2)
        shape_norm = self.norm(shape)
        texture_norm = self.norm(texture)

        shape_feature = shape_norm.view(shape_norm.size()[0], -1)
        texture_feature = texture_norm.view(texture_norm.size()[0], -1)

        shape_d0 = self.shape_conv1(self.shape_up1(shape_norm))
        shape_d1 = self.shape_conv2(self.shape_up2(shape_d0))
        shape_out = self.sigmoid(self.shape_out(shape_d1))

        texture_d0 = self.texture_conv1(self.texture_up1(torch.cat([texture_norm, shape_norm], dim=1)))
        texture_d1 = self.texture_conv2(self.texture_up2(texture_d0))
        texture_out = self.sigmoid(self.texture_out(texture_d1))

        return shape_feature, texture_feature, shape_out, texture_out

    pass


class EmbeddingNetCIFARSmallNorm3(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train

        self.input_size = 6
        self.embedding_size = 16
        self.has_bn = False

        # input 24
        self.conv11 = ConvBlock(3, 64, has_bn=self.has_bn)  # 6
        self.conv12 = ConvBlock(64, 64, has_bn=self.has_bn)  # 3
        self.pool1 = nn.MaxPool2d(2, 2)  # 3
        self.conv21 = ConvBlock(64, 128, has_bn=self.has_bn)  # 3
        self.conv22 = ConvBlock(128, 128, has_bn=self.has_bn)  # 3

        self.conv_shape1 = ConvBlock(128, 128, padding=0, has_bn=self.has_bn)  # 1
        self.conv_shape2 = ConvBlock(128, self.embedding_size, padding=0, ks=1, has_bn=self.has_bn, bias=False)  # 1
        self.conv_texture1 = ConvBlock(128, 128, padding=0, has_bn=self.has_bn, bias=self.has_bn)
        self.conv_texture2 = ConvBlock(128, self.embedding_size, padding=0, ks=1, has_bn=self.has_bn, bias=False)

        self.shape_conv1 = ConvBlock(self.embedding_size, 32, padding=0, ks=1, has_bn=self.has_bn)  # 3
        self.shape_up1 = nn.UpsamplingBilinear2d(scale_factor=3)  # 3
        self.shape_conv2 = ConvBlock(32, 32, has_bn=self.has_bn)  # 3
        self.shape_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.shape_conv3 = ConvBlock(32, 32, has_bn=self.has_bn)  # 6
        self.shape_out = ConvBlock(32, 1, has_relu=False, has_bn=self.has_bn)  # 6

        self.texture_conv1 = ConvBlock(self.embedding_size * 2, 128, padding=0, ks=1, has_bn=self.has_bn)  # 3
        self.texture_up1 = nn.UpsamplingBilinear2d(scale_factor=3) # 3
        self.texture_conv21 = ConvBlock(128, 128, has_bn=self.has_bn)  # 3
        self.texture_conv22 = ConvBlock(128, 128, has_bn=self.has_bn)  # 3
        self.texture_up2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 6
        self.texture_conv31 = ConvBlock(128, 64, has_bn=self.has_bn)  # 6
        self.texture_conv32 = ConvBlock(64, 64, has_bn=self.has_bn)  # 6
        self.texture_out = ConvBlock(64, 3, has_relu=False, has_bn=self.has_bn)  # 6

        self.norm = Normalize()
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        return self.forward_train(x) if self.is_train else self.forward_inference(x)

    def forward_train(self, x):
        e1 = self.pool1(self.conv12(self.conv11(x)))
        e2 = self.conv22(self.conv21(e1))

        shape_norm = self.norm(self.conv_shape2(self.conv_shape1(e2)))
        texture_norm = self.norm(self.conv_texture2(self.conv_texture1(e2)))

        shape_feature = shape_norm.view(shape_norm.size()[0], -1)
        texture_feature = texture_norm.view(texture_norm.size()[0], -1)

        shape_d1 = self.shape_up1(self.shape_conv1(shape_norm))
        shape_d2 = self.shape_up2(self.shape_conv2(shape_d1))
        shape_out = self.shape_out(self.shape_conv3(shape_d2))
        # shape_out = self.sigmoid(shape_out)

        texture_d0 = torch.cat([texture_norm, shape_norm], dim=1)
        texture_d1 = self.texture_up1(self.texture_conv1(texture_d0))
        texture_d2 = self.texture_up2(self.texture_conv22(self.texture_conv21(texture_d1)))
        texture_out = self.texture_out(self.texture_conv32(self.texture_conv31(texture_d2)))
        # texture_out = self.sigmoid(texture_out)

        return shape_feature, texture_feature, shape_out, texture_out

    def forward_inference(self, x):
        e1 = self.pool1(self.conv12(self.conv11(x)))
        e2 = self.conv22(self.conv21(e1))

        shape_norm = self.norm(self.conv_shape2(self.conv_shape1(e2)))
        texture_norm = self.norm(self.conv_texture2(self.conv_texture1(e2)))

        shape_feature = shape_norm.view(shape_norm.size()[0], -1)
        texture_feature = texture_norm.view(texture_norm.size()[0], -1)
        return shape_feature, texture_feature

    pass


class Runner(object):

    def __init__(self, root_ckpt_dir, model=EmbeddingNetCIFAR,
                 data_root_path='/mnt/4T/Data/cifar/cifar-10', use_gpu=False, gpu_id="0"):
        self.root_ckpt_dir = root_ckpt_dir
        self.device = self.gpu_setup(use_gpu, gpu_id)

        self.model = model().to(self.device)
        Tools.print("Total param: {}".format(self.view_model_param(self.model)))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=0.5, patience=0, verbose=True)

        self.loss_shape = nn.MSELoss()
        self.loss_texture = nn.MSELoss()
        self.data_CIFAR10 = DataCIFAR10(train_batch=64, test_batch=32, data_root_path=data_root_path)
        pass

    def train(self, epochs=100):
        for epoch in range(epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            epoch_loss, epoch_loss_shape, epoch_loss_texture = self.train_epoch(self.data_CIFAR10.train_loader)
            self.save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            self.scheduler.step(epoch_loss)

            Tools.print("lr={:.4f}, loss={:.4f}, shape={:.4f}, texture={:.4f}".format(
                self.optimizer.param_groups[0]['lr'], epoch_loss, epoch_loss_shape, epoch_loss_texture))
            pass
        pass

    def train_epoch(self, data_loader):
        self.model.train()

        epoch_loss, epoch_loss_shape, epoch_loss_texture = 0, 0, 0
        for i, (img, target) in enumerate(data_loader):
            start = time.time()
            net_input, sp_info = self.get_super_pixel(img.detach().numpy())
            super_pixel_time = time.time() - start

            start = time.time()
            batch_img = torch.from_numpy(net_input["data"]).float().to(self.device)
            shape_target = torch.from_numpy(net_input["shape"]).float().to(self.device)
            self.optimizer.zero_grad()
            shape_feature, texture_feature, shape_out, texture_out = self.model.forward(batch_img)
            loss_shape = self.loss_shape(shape_out, shape_target)
            _positions = batch_img >= 0
            loss_texture = self.loss_texture(texture_out[_positions], batch_img[_positions])
            loss = loss_shape + loss_texture
            loss.backward()
            self.optimizer.step()
            net_time = time.time() - start

            now_loss = loss.detach().item()
            now_loss_shape = loss_shape.detach().item()
            now_loss_texture = loss_texture.detach().item()
            epoch_loss += now_loss
            epoch_loss_shape += now_loss_shape
            epoch_loss_texture += now_loss_texture

            if i % 100 == 0:
                Tools.print("{}-{} time:{:4f} {:4f}, loss={:4f}/{:4f} - {:4f}/{:4f} - {:4f}/{:4f}".format(
                    i, len(data_loader), super_pixel_time, net_time, now_loss, epoch_loss / (i+1),
                    now_loss_shape, epoch_loss_shape / (i+1), now_loss_texture, epoch_loss_texture / (i+1)))
                pass

            pass

        return epoch_loss/len(data_loader), epoch_loss_shape/len(data_loader), epoch_loss_texture/len(data_loader)

    @staticmethod
    def get_super_pixel(batch_img, ds_image_size=32, super_pixel_size=4, super_pixel_data_size=6):
        now_data_list, now_shape_list = [], []
        now_segment_list, now_super_pixel_info_list, now_adjacency_info_list = [], [], []
        for img in batch_img:
            deal_super_pixel = DealSuperPixel(image_data=img,
                                              ds_image_size=ds_image_size, super_pixel_size=super_pixel_size)
            now_segment, now_super_pixel_info, now_adjacency_info = deal_super_pixel.run()
            for key in now_super_pixel_info:
                now_data = now_super_pixel_info[key]["data2"] / 255
                now_data = cv2.resize(now_data, (super_pixel_data_size,
                                                 super_pixel_data_size), interpolation=cv2.INTER_NEAREST)
                now_shape = now_super_pixel_info[key]["label"] / 1
                now_shape = cv2.resize(now_shape, (super_pixel_data_size,
                                                   super_pixel_data_size), interpolation=cv2.INTER_NEAREST)
                now_shape = np.expand_dims(now_shape, axis=-1)
                now_data_list.append(now_data)
                now_shape_list.append(now_shape)
                pass

            now_segment_list.append(now_segment)
            now_super_pixel_info_list.append(now_super_pixel_info)
            now_adjacency_info_list.append(now_adjacency_info)
            pass

        net_input = {"data": np.transpose(now_data_list, axes=(0, 3, 1, 2)),
                     "shape": np.transpose(now_shape_list, axes=(0, 3, 1, 2))}
        sp_info = {"segment": now_segment_list, "node": now_super_pixel_info_list, "edge": now_adjacency_info_list}
        return net_input, sp_info

    @staticmethod
    def save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    def load_model(self, model_file_name):
        self.model.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print("restore from {}".format(model_file_name))
        pass

    @staticmethod
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

    @staticmethod
    def view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


class VisualEmbeddingVisualization(object):

    def __init__(self, model_file_name, model=EmbeddingNetCIFAR,
                 data_root_path='/mnt/4T/Data/cifar/cifar-10', use_gpu=False, gpu_id="0"):
        self.model_file_name = model_file_name
        self.device = Runner.gpu_setup(use_gpu, gpu_id)

        self.data_CIFAR10 = DataCIFAR10(train_batch=64, test_batch=32, data_root_path=data_root_path)

        self.model = model().to(self.device)
        self.model.load_state_dict(torch.load(self.model_file_name), strict=False)

        Tools.print("Total param: {}, Restore from {}".format(
            Runner.view_model_param(self.model), self.model_file_name))
        pass

    def show_train(self):
        self.model.eval()

        for i, (img, target) in enumerate(self.data_CIFAR10.train_loader):
            net_input, sp_info = self.get_super_pixel(img.detach().numpy())
            batch_img = torch.from_numpy(net_input["data"]).float().to(self.device)
            shape_target = torch.from_numpy(net_input["shape"]).float().to(self.device)

            self.optimizer.zero_grad()
            shape_feature, texture_feature, shape_out, texture_out = self.model.forward(batch_img)

            for sp_i in range(batch_img.size(0)):
                # 1
                Image.fromarray(np.asarray(np.transpose(batch_img.detach().numpy()[sp_i],
                                                        axes=(1, 2, 0)) * 255, dtype=np.uint8)).resize((60, 60)).show()
                # 2
                Image.fromarray(shape_target.detach().numpy()[sp_i][0] * 255).resize((60, 60)).show()
                # 3
                texture_out_one = np.transpose(texture_out.detach().numpy()[sp_i], axes=(1, 2, 0))
                texture_out_one[texture_out_one < 0] = 0
                texture_out_one[texture_out_one > 1] = 1
                Image.fromarray(np.asarray(texture_out_one * 255, dtype=np.uint8)).resize((60, 60)).show()
                # 4
                shape_out_one = shape_out.detach().numpy()[sp_i][0]
                shape_out_one[shape_out_one < 0] = 0
                shape_out_one[shape_out_one > 1] = 1
                Image.fromarray(np.asarray(shape_out_one * 255, dtype=np.uint8)).resize((60, 60)).show()
                pass
            pass

        pass

    def reconstruct_image(self):
        self.model.eval()

        for i, (img, target) in enumerate(self.data_CIFAR10.test_loader):
            img = img.detach().numpy()

            net_input, sp_info = Runner.get_super_pixel(img)
            batch_img = torch.from_numpy(net_input["data"]).float().to(self.device)

            sp_node = sp_info["node"]
            shape_feature, texture_feature, shape_out, texture_out = self.model.forward(batch_img)
            shape_out = shape_out.detach().numpy()
            texture_out = texture_out.detach().numpy()

            node_num = 0
            for img_i in range(len(img)):
                now_img = img[img_i]
                now_node = sp_node[img_i]
                now_texture = np.transpose(texture_out[node_num: node_num + len(now_node)], axes=(0, 2, 3, 1))
                now_shape = np.transpose(shape_out[node_num: node_num + len(now_node)], axes=(0, 2, 3, 1))

                now_result = np.zeros_like(now_img, dtype=np.float)
                for sp_i in range(len(now_texture)):
                    now_area_sp_i = now_node[sp_i]["area"]
                    # 形状
                    # now_shape_sp_i = np.squeeze(now_node[sp_i]["label"], axis=-1)
                    now_shape_sp_i = cv2.resize(now_shape[sp_i][:, :, 0], (now_area_sp_i[3] - now_area_sp_i[2] + 1,
                                                                           now_area_sp_i[1] - now_area_sp_i[0] + 1),
                                                interpolation=cv2.INTER_NEAREST)
                    now_shape_sp_i[now_shape_sp_i < 0.2] = 0

                    # 纹理
                    now_texture_sp_i = cv2.resize(now_texture[sp_i], (now_area_sp_i[3] - now_area_sp_i[2] + 1,
                                                                      now_area_sp_i[1] - now_area_sp_i[0] + 1),
                                                  interpolation=cv2.INTER_NEAREST)
                    now_texture_sp_i[now_texture_sp_i > 1] = 1
                    now_texture_sp_i[now_texture_sp_i < 0] = 0

                    # 填充
                    _result_area = now_result[now_area_sp_i[0]: now_area_sp_i[1] + 1,
                                   now_area_sp_i[2]: now_area_sp_i[3] + 1, :]
                    _result_area[now_shape_sp_i > 0] = now_texture_sp_i[now_shape_sp_i > 0]
                    pass

                Image.fromarray(now_img).show()
                Image.fromarray(np.asarray(now_result * 255, dtype=np.uint8)).show()
                node_num += len(now_node)
                pass

            pass

        pass

    pass


class VisualEmbedding(object):

    def __init__(self, model_file_name, model=EmbeddingNetCIFAR,
                 data_root_path='/mnt/4T/Data/cifar/cifar-10', use_gpu=False, gpu_id="0"):
        self.model_file_name = model_file_name
        self.device = Runner.gpu_setup(use_gpu, gpu_id)

        self.data_CIFAR10 = DataCIFAR10(train_batch=64, test_batch=32, data_root_path=data_root_path)

        self.model = model().to(self.device)
        self.model.load_state_dict(torch.load(self.model_file_name), strict=False)

        Tools.print("Total param: {}, Restore from {}".format(
            Runner.view_model_param(self.model), self.model_file_name))
        pass

    def run(self):
        self.model.eval()
        for i, (img, target) in enumerate(self.data_CIFAR10.train_loader):
            img = img.detach().numpy()

            start = time.time()
            sp_info = self.get_sp_info(img)
            super_pixel_time = time.time() - start

            Tools.print("{} Time: {}".format(i, super_pixel_time))
            pass
        pass

    def get_sp_info(self, img):
        net_input, sp_info = Runner.get_super_pixel(img)
        batch_img = torch.from_numpy(net_input["data"]).float().to(self.device)

        shape_feature, texture_feature, _, _ = self.model.forward(batch_img)
        shape_feature, texture_feature = shape_feature.detach().numpy(), texture_feature.detach().numpy()

        node_num = 0
        for img_i in range(len(img)):
            now_node_i = sp_info["node"][img_i]
            now_shape_feature_i = shape_feature[node_num: node_num + len(now_node_i)]
            now_texture_feature_i = texture_feature[node_num: node_num + len(now_node_i)]

            for sp_i in range(len(now_node_i)):
                now_node_i[sp_i]["feature_shape"] = now_shape_feature_i[sp_i]
                now_node_i[sp_i]["feature_texture"] = now_texture_feature_i[sp_i]
                pass

            node_num += len(now_node_i)
            pass
        return sp_info

    pass


if __name__ == '__main__':
    ############################################################################################
    # runner = Runner(root_ckpt_dir=Tools.new_dir("ckpt\\norm4_sigmoid"),
    #                 data_root_path='D:\data\CIFAR', model=EmbeddingNetCIFARSmallNorm3)
    # runner = Runner(root_ckpt_dir=Tools.new_dir("./ckpt/norm4"),
    #                 model=EmbeddingNetCIFARSmallNorm3, use_gpu=True, gpu_id="0")
    # runner.load_model("ckpt\\norm3\\epoch_1.pkl")
    # runner.train(50)
    ############################################################################################
    # visual_embedding_visualization = VisualEmbeddingVisualization(
    #     model_file_name="ckpt\\norm\\epoch_1.pkl", model=EmbeddingNetCIFARSmallNorm)
    # visual_embedding_visualization.show_train()
    ############################################################################################
    visual_embedding_visualization = VisualEmbeddingVisualization(
        model_file_name="./ckpt/norm4/epoch_7.pkl", model=EmbeddingNetCIFARSmallNorm3)
    visual_embedding_visualization.reconstruct_image()
    ############################################################################################
    # visual_embedding = VisualEmbedding(model_file_name="ckpt\\norm\\epoch_1.pkl", model=EmbeddingNetCIFARSmallNorm)
    # visual_embedding.run()
    ############################################################################################
    pass
