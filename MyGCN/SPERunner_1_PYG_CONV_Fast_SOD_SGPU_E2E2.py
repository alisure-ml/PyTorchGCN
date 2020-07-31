import os
import cv2
import glob
import torch
import random
import skimage
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
from torchvision.models import vgg13_bn
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


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


class DealSuperPixel(object):

    def __init__(self, image_data, label_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))
        self.label_data = label_data if len(label_data) == self.ds_image_size else cv2.resize(
            label_data, (self.ds_image_size, self.ds_image_size))

        try:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        except TypeError:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter)
            pass

        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        edge_index, sp_label, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            where = self.segment == i
            # 计算标签
            label = np.mean(self.label_data[where])
            sp_label.append(label)

            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(where, selem=skimage.morphology.square(3))
            edge_index.extend([[i, sp_id] for sp_id in np.unique(self.segment[_now_adj]) if sp_id != i])

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            pixel_edge_index = np.argwhere(_dis <= 2)
            pixel_edge_w = np.ones(len(pixel_edge_index))
            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w, label])
            pass

        sp_adj = np.asarray(edge_index)
        sp_label = np.asarray(sp_label)
        return self.segment, sp_adj, pixel_adj, sp_label

    pass


class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img, 'label': mask}

    pass


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}

    pass


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        self.size = (int(size), int(size))
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        i, j, h, w = self.get_params(img, self.size)
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return {'image': img, 'label': mask}

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path, down_ratio=4, down_ratio2=2, is_train=True,
                 image_size_train=224, image_size_test=256, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size_train if self.is_train else image_size_test
        self.image_size_for_sp = self.image_size // down_ratio
        self.label_size_for_sod = self.image_size // down_ratio2
        self.data_root_path = data_root_path

        # 路径
        self.data_image_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Image" if self.is_train else "DUTS-TE-Image")
        self.data_label_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Mask" if self.is_train else "DUTS-TE-Mask")

        # 数据增强
        self.transform_train = transforms.Compose([FixedResize(image_size_test),
                                                   RandomHorizontalFlip(), RandomCrop(self.image_size)])
        self.transform_test = transforms.Compose([FixedResize(self.image_size)])

        # 准备数据
        self.image_name_list, self.label_name_list = self.get_image_label_name()
        pass

    def get_image_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_image_path, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_label_path, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        return tra_img_name_list, tra_lbl_name_list

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        label = Image.open(self.label_name_list[idx])
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_name = self.image_name_list[idx]

        # 数据增强
        sample = {'image': image, 'label': label}
        sample = self.transform_train(sample) if self.is_train else self.transform_test(sample)
        image, label = sample['image'], sample['label']
        label_for_sod = np.asarray(label.resize((self.label_size_for_sod, self.label_size_for_sod))) / 255

        # 归一化
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

        # 超像素
        image_small_data = np.asarray(image.resize((self.image_size_for_sp, self.image_size_for_sp)))
        label_for_sp = np.asarray(label.resize((self.image_size_for_sp, self.image_size_for_sp))) / 255
        graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_for_sp)

        # 返回
        return graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name

    def get_sp_info(self, image, label):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=image, label_data=label,
                                          ds_image_size=self.image_size_for_sp, super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj, sp_label = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = Data(edge_index=torch.from_numpy(np.transpose(sp_adj, axes=(1, 0))),
                     num_nodes=len(pixel_adj), y=torch.from_numpy(sp_label).float(), num_sp=len(pixel_adj))
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([super_pixel[3]]),
                               edge_w=torch.from_numpy(super_pixel[2]).unsqueeze(1).float())
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph, segment

    @staticmethod
    def collate_fn(samples):
        graphs, pixel_graphs, images, labels_sp, labels_sod, segments, images_small, image_name = map(list,
                                                                                                      zip(*samples))

        images = torch.cat(images)
        images_small = torch.tensor(images_small)

        # 超像素图
        batched_graph = Batch.from_data_list(graphs)

        # 像素图
        _pixel_graphs = []
        for super_pixel_i, pixel_graph in enumerate(pixel_graphs):
            for now_graph in pixel_graph:
                now_graph.data_where[:, 0] = super_pixel_i
                _pixel_graphs.append(now_graph)
            pass
        batched_pixel_graph = Batch.from_data_list(_pixel_graphs)

        return images, labels_sp, labels_sod, batched_graph, batched_pixel_graph, segments, images_small, image_name

    pass


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, ks=3, has_relu=True, has_bn=True, bias=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
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


class CONVNet(nn.Module):

    def __init__(self, layer_num=20):  # 14, 20
        super().__init__()
        self.features = vgg13_bn(pretrained=True).features[0: layer_num]
        self.out_num2 = self.features[-2].num_features
        self.out_num1 = self.out_num2 // 2
        pass

    def forward(self, x):
        block_1 = self.features[0: 6](x)
        block_1_max = self.features[6](block_1)
        block_2 = self.features[7: 13](block_1_max)
        block_2_max = self.features[13](block_2)
        block_3 = self.features[14: 20](block_2_max)
        assert self.out_num1 == block_2.size(1)
        assert self.out_num2 == block_3.size(1)
        return block_2, block_3

    pass


class GCNNet1(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.improved = improved
        self.out_num = self.hidden_dims[-1]

        # self.embedding_h = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass
        pass

    def forward(self, data):
        # hidden_nodes_feat = self.embedding_h(data.x)
        hidden_nodes_feat = data.x
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_mean_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], skip_which=[1, 2, 3],
                 skip_dim=128, sout=1, has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.improved = improved
        self.out_num = len(skip_which) * skip_dim

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass

        # skip
        self.skip_connect_index = skip_which
        self.skip_connect_list = nn.ModuleList()
        self.skip_connect_bn_list = nn.ModuleList()
        for hidden_dim in [self.hidden_dims[which-1] for which in skip_which]:
            self.skip_connect_list.append(nn.Linear(hidden_dim, skip_dim, bias=False))
            # self.skip_connect_list.append(GCNConv(hidden_dim, skip_dim,
            #                                       normalize=self.normalize, improved=self.improved))
            self.skip_connect_bn_list.append(nn.BatchNorm1d(skip_dim))
            pass

        self.readout_mlp1 = nn.Linear(self.out_num, self.out_num, bias=False)
        self.readout_mlp2 = nn.Linear(self.out_num, sout, bias=False)
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        gcn_hidden_nodes_feat = [hidden_nodes_feat]
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)
            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat

            gcn_hidden_nodes_feat.append(hidden_nodes_feat)
            pass

        skip_connect = []
        for sc, index, bn in zip(self.skip_connect_list, self.skip_connect_index, self.skip_connect_bn_list):
            # Conv
            # sc_feat = sc(gcn_hidden_nodes_feat[index], data.edge_index)
            sc_feat = sc(gcn_hidden_nodes_feat[index])
            if self.has_bn:
                sc_feat = bn(sc_feat)
            sc_feat = self.relu(sc_feat)

            skip_connect.append(sc_feat)
            pass

        out_feat = torch.cat(skip_connect, dim=1)
        out_mlp = self.readout_mlp1(out_feat)
        out_mlp = self.relu(out_mlp)
        logits = self.readout_mlp2(out_mlp).view(-1)
        return out_feat, logits, torch.sigmoid(logits)

    pass


class SODNet(nn.Module):

    def __init__(self, conv1_feature_num, conv2_feature_num, sod_gcn1_feature_num,
                 sod_gcn2_feature_num, mout=128, sout=1):
        super().__init__()
        self.conv_conv1 = ConvBlock(conv1_feature_num, cout=mout, ks=3)
        self.conv_conv2 = ConvBlock(conv2_feature_num, cout=mout, ks=3)
        self.conv_sod_gcn1 = ConvBlock(sod_gcn1_feature_num, cout=mout, ks=3)
        self.conv_sod_gcn2 = ConvBlock(sod_gcn2_feature_num, cout=mout, ks=3)

        self.cat_conv1 = ConvBlock(mout * 2, cout=mout, ks=3)
        self.cat_conv2 = ConvBlock(mout * 2, cout=mout, ks=3)
        self.cat_sod_gcn = ConvBlock(mout * 2, cout=mout, ks=3)

        self.conv_final_1 = ConvBlock(mout, cout=mout, ks=3)
        self.conv_final_2 = ConvBlock(mout, cout=sout, ks=3, has_relu=False, has_bn=False, bias=False)
        pass

    def forward(self, conv1_feature, conv2_feature, sod_gcn1_feature, sod_gcn2_feature):
        conv_sod_gcn2 = self.conv_sod_gcn2(sod_gcn2_feature)  # 256, 56
        conv_sod_gcn1 = self.conv_sod_gcn1(sod_gcn1_feature)  # 256, 56
        cat_sod_gcn = torch.cat([conv_sod_gcn1, conv_sod_gcn2], dim=1)  # 512, 56
        cat_sod_gcn = self.cat_sod_gcn(cat_sod_gcn)  # 256, 56

        conv_conv2 = self.conv_conv2(conv2_feature)
        cat_conv2 = torch.cat([cat_sod_gcn, conv_conv2], dim=1)  # 512, 56
        cat_conv2 = self.cat_conv2(cat_conv2)  # 256, 56
        cat_conv2 = F.interpolate(cat_conv2, conv1_feature.size()[2:], mode='bilinear', align_corners=True)  # 256, 112

        conv_conv1 = self.conv_conv1(conv1_feature)
        cat_conv1 = torch.cat([cat_conv2, conv_conv1], dim=1)  # 512, 112
        cat_conv1 = self.cat_conv1(cat_conv1)  # 256, 112

        out_feat = cat_conv1
        out_feat = self.conv_final_1(out_feat)  # 256, 112
        out_feat = self.conv_final_2(out_feat)  # 1, 112
        return out_feat, torch.sigmoid(out_feat)

    pass


class SODNet2(nn.Module):

    def __init__(self, conv1_feature_num, conv2_feature_num, sod_gcn1_feature_num,
                 sod_gcn2_feature_num, mout=128, sout=1):
        super().__init__()
        self.conv_conv1_1 = ConvBlock(conv1_feature_num, cout=conv1_feature_num, ks=3)
        self.conv_conv1_2 = ConvBlock(conv1_feature_num, cout=mout, ks=3)
        self.conv_conv2_1 = ConvBlock(conv2_feature_num, cout=conv2_feature_num, ks=3)
        self.conv_conv2_2 = ConvBlock(conv2_feature_num, cout=mout, ks=3)
        self.conv_sod_gcn1_1 = ConvBlock(sod_gcn1_feature_num, cout=sod_gcn1_feature_num, ks=3)
        self.conv_sod_gcn1_2 = ConvBlock(sod_gcn1_feature_num, cout=mout, ks=3)
        self.conv_sod_gcn2_1 = ConvBlock(sod_gcn2_feature_num, cout=sod_gcn2_feature_num, ks=3)
        self.conv_sod_gcn2_2 = ConvBlock(sod_gcn2_feature_num, cout=mout, ks=3)

        self.cat_conv1_1 = ConvBlock(mout * 2, cout=mout * 2, ks=3)
        self.cat_conv1_2 = ConvBlock(mout * 2, cout=mout, ks=3)
        self.cat_conv2_1 = ConvBlock(mout * 2, cout=mout * 2, ks=3)
        self.cat_conv2_2 = ConvBlock(mout * 2, cout=mout, ks=3)
        self.cat_sod_gcn_1 = ConvBlock(mout * 2, cout=mout * 2, ks=3)
        self.cat_sod_gcn_2 = ConvBlock(mout * 2, cout=mout, ks=3)

        self.conv_final_1 = ConvBlock(mout, cout=mout, ks=3)
        self.conv_final_2 = ConvBlock(mout, cout=sout, ks=3, has_relu=False, has_bn=False, bias=False)
        pass

    def forward(self, conv1_feature, conv2_feature, sod_gcn1_feature, sod_gcn2_feature):
        conv_sod_gcn2 = self.conv_sod_gcn2_2(self.conv_sod_gcn2_1(sod_gcn2_feature))  # 256, 56
        conv_sod_gcn1 = self.conv_sod_gcn1_2(self.conv_sod_gcn1_1(sod_gcn1_feature))  # 256, 56
        cat_sod_gcn = torch.cat([conv_sod_gcn1, conv_sod_gcn2], dim=1)  # 512, 56
        cat_sod_gcn = self.cat_sod_gcn_2(self.cat_sod_gcn_1(cat_sod_gcn))  # 256, 56

        conv_conv2 = self.conv_conv2_2(self.conv_conv2_1(conv2_feature))
        cat_conv2 = torch.cat([cat_sod_gcn, conv_conv2], dim=1)  # 512, 56
        cat_conv2 = self.cat_conv2_2(self.cat_conv2_1(cat_conv2))  # 256, 56
        cat_conv2 = F.interpolate(cat_conv2, conv1_feature.size()[2:], mode='bilinear', align_corners=True)  # 256, 112

        conv_conv1 = self.conv_conv1_2(self.conv_conv1_1(conv1_feature))
        cat_conv1 = torch.cat([cat_conv2, conv_conv1], dim=1)  # 512, 112
        cat_conv1 = self.cat_conv1_2(self.cat_conv1_1(cat_conv1))  # 256, 112

        out_feat = cat_conv1
        out_feat = self.conv_final_1(out_feat)  # 256, 112
        out_feat = self.conv_final_2(out_feat)  # 1, 112
        return out_feat, torch.sigmoid(out_feat)

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=20, has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        assert conv_layer_num == 20
        self.model_conv = CONVNet(layer_num=conv_layer_num)  # 20
        self.model_gnn1 = GCNNet1(in_dim=self.model_conv.out_num2, hidden_dims=[256, 256],
                                  has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
        self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[512, 512, 1024, 1024],
                                  skip_which=[2, 4], skip_dim=256, has_bn=has_bn,
                                  normalize=normalize, residual=residual, improved=improved)

        self.model_sod = SODNet2(conv1_feature_num=self.model_conv.out_num1,
                                conv2_feature_num=self.model_conv.out_num2,
                                sod_gcn1_feature_num=self.model_gnn1.out_num,
                                sod_gcn2_feature_num=self.model_gnn2.out_num, mout=256, sout=1)
        pass

    def forward(self, images, batched_graph, batched_pixel_graph):
        # model 1
        conv_feature = self.model_conv(images)  # 128, 64; 256, 64
        conv1_feature, conv2_feature = conv_feature

        # model 2
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = conv2_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph)
        # 构造特征 for SOD
        sod_gcn1_feature = self.sod_feature(data_where, gcn1_feature, batched_pixel_graph=batched_pixel_graph)

        # model 3
        batched_graph.x = gcn1_feature
        gcn2_feature, gcn2_logits, gcn2_logits_sigmoid = self.model_gnn2.forward(batched_graph)
        # 构造特征 for SOD
        sod_gcn2_feature = self.sod_feature(data_where, gcn2_feature, batched_pixel_graph=batched_pixel_graph)

        # SOD
        sod_logits, sod_logits_sigmoid = self.model_sod.forward(
            conv1_feature, conv2_feature, sod_gcn1_feature, sod_gcn2_feature)
        return gcn2_logits, gcn2_logits_sigmoid, sod_logits, sod_logits_sigmoid

    @staticmethod
    def sod_feature(data_where, gcn_feature, batched_pixel_graph):
        # 构造特征
        _shape = torch.max(data_where, dim=0)[0] + 1
        _size = (_shape[0], gcn_feature.shape[-1], _shape[1], _shape[2])
        _gcn_feature_for_sod = gcn_feature[batched_pixel_graph.batch]

        sod_gcn_feature = torch.Tensor(size=_size).to(gcn_feature.device)
        sod_gcn_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]] = _gcn_feature_for_sod
        return sod_gcn_feature

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path, down_ratio=4, batch_size=64, image_size_train=224, image_size_test=256,
                 sp_size=4, train_print_freq=100, test_print_freq=50, root_ckpt_dir="./ckpt2/norm3", lr=None,
                 num_workers=8, use_gpu=True, gpu_id="1", conv_layer_num=14, has_mask=False,
                 has_bn=True, normalize=True, residual=False, improved=False, weight_decay=0.0, is_sgd=False):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(
            data_root_path=data_root_path, is_train=True, down_ratio=down_ratio,
            image_size_train=image_size_train, image_size_test=image_size_test, sp_size=sp_size)
        self.test_dataset = MyDataset(
            data_root_path=data_root_path, is_train=False, down_ratio=down_ratio,
            image_size_train=image_size_train, image_size_test=image_size_test, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=conv_layer_num, has_bn=has_bn,
                              normalize=normalize, residual=residual, improved=improved).to(self.device)

        if is_sgd:
            # self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]]
            self.lr_s = [[0, 0.01], [50, 0.001], [90, 0.0001]] if lr is None else lr
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][1],
                                             momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]] if lr is None else lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=weight_decay)

        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))

        self.has_mask = has_mask
        self.loss_class = nn.BCELoss().to(self.device)
        pass

    def loss_bce(self, logits_sigmoid, labels):
        if self.has_mask:
            positions = (labels > 0.9) + (labels < 0.1)
            loss = self.loss_class(logits_sigmoid[positions], labels[positions])
        else:
            loss = self.loss_class(logits_sigmoid, labels)
        return loss

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)

        # keys = [c for c in ckpt if "model_gnn1.gcn_list.0" in c]
        # for c in keys:
        #     del ckpt[c]
        #     Tools.print(c)
        #     pass

        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            (train_loss, train_loss1, train_loss2, train_mae, train_score,
             train_mae2, train_score2, train_mae3, train_score3, train_mae4, train_score4) = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            (test_loss, test_loss1, test_loss2, test_mae, test_score,
             test_mae2, test_score2, test_mae3, test_score3, test_mae4, test_score4) = self.test()

            Tools.print('E:{:2d}, Train sod-mae-score={:.4f}-{:.4f} gcn-mae-score={:.4f}-{:.4f} '
                        'gcn-final-mse-score={:.4f}-{:.4f}({:.4f}/{:.4f}) loss={:.4f}({:.4f}+{:.4f})'.format(
                epoch, train_mae4, train_score4, train_mae, train_score,
                train_mae2, train_score2, train_mae3, train_score3, train_loss, train_loss1, train_loss2))
            Tools.print('E:{:2d}, Test  sod-mae-score={:.4f}-{:.4f} gcn-mae-score={:.4f}-{:.4f} '
                        'gcn-final-mse-score={:.4f}-{:.4f}({:.4f}/{:.4f}) loss={:.4f}({:.4f}+{:.4f})'.format(
                epoch, test_mae4, test_score4, test_mae, test_score,
                test_mae2, test_score2, test_mae3, test_score3, test_loss, test_loss1, test_loss2))
            pass
        pass

    def _train_epoch(self):
        self.model.train()

        # 统计
        th_num = 25
        epoch_loss, epoch_loss1, epoch_loss2, nb_data = 0, 0, 0, 0
        epoch_mae, epoch_prec, epoch_recall = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae2, epoch_prec2, epoch_recall2 = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae3, epoch_prec3, epoch_recall3 = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae4, epoch_prec4, epoch_recall4 = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        for i, (images, labels_sp, labels_sod, batched_graph,
                batched_pixel_graph, segments, _, _) in enumerate(self.train_loader):
            # Run
            self.optimizer.zero_grad()

            # Data
            images = images.float().to(self.device)
            labels = batched_graph.y.to(self.device)
            labels_sp = torch.unsqueeze(torch.Tensor(labels_sp), dim=1).to(self.device)
            labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)
            batched_graph.batch = batched_graph.batch.to(self.device)
            batched_graph.edge_index = batched_graph.edge_index.to(self.device)

            batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
            batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
            batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

            _, gcn_logits_sigmoid, _, sod_logits_sigmoid = self.model.forward(images, batched_graph,
                                                                              batched_pixel_graph)

            loss1 = self.loss_bce(gcn_logits_sigmoid, labels)
            loss2 = self.loss_class(sod_logits_sigmoid, labels_sod)
            loss = loss1 + loss2

            loss.backward()
            self.optimizer.step()

            labels_val = labels.cpu().detach().numpy()
            labels_sp_val = labels_sp.cpu().detach().numpy()
            labels_sod_val = labels_sod.cpu().detach().numpy()
            gcn_logits_sigmoid_val = gcn_logits_sigmoid.cpu().detach().numpy()
            sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

            # Stat
            nb_data += images.size(0)
            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()

            # cal 1
            mae = self._eval_mae(gcn_logits_sigmoid_val, labels_val)
            prec, recall = self._eval_pr(gcn_logits_sigmoid_val, labels_val, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # cal 2
            cum_add = np.cumsum([0] + batched_graph.num_sp.tolist())
            for one in range(len(segments)):
                tar_sod = self._cal_sod(labels[cum_add[one]: cum_add[one + 1]].tolist(), segments[one])
                pre_sod = self._cal_sod(gcn_logits_sigmoid_val[cum_add[one]: cum_add[one + 1]].tolist(), segments[one])

                mae2 = self._eval_mae(pre_sod, tar_sod)
                prec2, recall2 = self._eval_pr(pre_sod, tar_sod, th_num)
                epoch_mae2 += mae2
                epoch_prec2 += prec2
                epoch_recall2 += recall2

                mae3 = self._eval_mae(pre_sod, labels_sp_val[one])
                prec3, recall3 = self._eval_pr(pre_sod, labels_sp_val[one], th_num)
                epoch_mae3 += mae3
                epoch_prec3 += prec3
                epoch_recall3 += recall3
                pass

            # cal 3
            mae4 = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
            prec4, recall4 = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
            epoch_mae4 += mae4
            epoch_prec4 += prec4
            epoch_recall4 += recall4

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}+{:.4f})-{:.4f}({:.4f}+{:.4f}) "
                            "sod-mse={:.4f}({:.4f}) gcn-mse={:.4f}({:.4f}) gcn-final-mse={:.4f}({:.4f})".format(
                    i, len(self.train_loader),
                    loss.detach().item(), loss1.detach().item(), loss2.detach().item(),
                    epoch_loss / (i + 1), epoch_loss1 / (i + 1), epoch_loss2 / (i + 1),
                    mae4, epoch_mae4 / (i + 1), mae, epoch_mae / (i + 1), epoch_mae2 / nb_data, epoch_mae3 / nb_data))
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_loss / len(self.train_loader), epoch_mae / len(self.train_loader)
        avg_loss1, avg_loss2 = epoch_loss1 / len(self.train_loader), epoch_loss2 / len(self.train_loader)
        avg_prec, avg_recall = epoch_prec / len(self.train_loader), epoch_recall / len(self.train_loader)
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        # 结果2
        avg_mae2, avg_prec2, avg_recall2 = epoch_mae2/nb_data, epoch_prec2/nb_data, epoch_recall2/nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)
        # 结果3
        avg_mae3, avg_prec3, avg_recall3 = epoch_mae3/nb_data, epoch_prec3/nb_data, epoch_recall3/nb_data
        score3 = (1 + 0.3) * avg_prec3 * avg_recall3 / (0.3 * avg_prec3 + avg_recall3)
        # 结果4
        avg_mae4 = epoch_mae4/len(self.train_loader)
        avg_prec4, avg_recall4 = epoch_prec4/len(self.train_loader), epoch_recall4/len(self.train_loader)
        score4 = (1 + 0.3) * avg_prec4 * avg_recall4 / (0.3 * avg_prec4 + avg_recall4)

        return (avg_loss, avg_loss1, avg_loss2,
                avg_mae, score.max(), avg_mae2, score2.max(), avg_mae3, score3.max(), avg_mae4, score4.max())

    def test(self, model_file=None, is_train_loader=False):
        if model_file:
            self.load_model(model_file_name=model_file)

        self.model.eval()

        Tools.print()
        th_num = 25

        # 统计
        epoch_test_loss, epoch_test_loss1, epoch_test_loss2, nb_data = 0, 0, 0, 0
        epoch_test_mae, epoch_test_mae2, epoch_test_mae3, epoch_test_mae4 = 0.0, 0.0, 0.0, 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_prec2, epoch_test_recall2 = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_prec3, epoch_test_recall3 = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_prec4, epoch_test_recall4 = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        loader = self.train_loader if is_train_loader else self.test_loader
        with torch.no_grad():
            for i, (images, labels_sp, labels_sod,
                    batched_graph, batched_pixel_graph, segments, _, _) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                labels = batched_graph.y.to(self.device)
                labels_sp = torch.unsqueeze(torch.Tensor(labels_sp), dim=1).to(self.device)
                labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                _, gcn_logits_sigmoid, _, sod_logits_sigmoid = self.model.forward(images, batched_graph,
                                                                                  batched_pixel_graph)

                loss1 = self.loss_bce(gcn_logits_sigmoid, labels)
                loss2 = self.loss_bce(sod_logits_sigmoid, labels_sod)
                loss = loss1 + loss2

                labels_val = labels.cpu().detach().numpy()
                labels_sp_val = labels_sp.cpu().detach().numpy()
                labels_sod_val = labels_sod.cpu().detach().numpy()
                gcn_logits_sigmoid_val = gcn_logits_sigmoid.cpu().detach().numpy()
                sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_loss1 += loss1.detach().item()
                epoch_test_loss2 += loss2.detach().item()

                # cal 1
                mae = self._eval_mae(gcn_logits_sigmoid_val, labels_val)
                prec, recall = self._eval_pr(gcn_logits_sigmoid_val, labels_val, th_num)
                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # cal 2
                cum_add = np.cumsum([0] + batched_graph.num_sp.tolist())
                for one in range(len(segments)):
                    tar_sod = self._cal_sod(labels[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_sod = self._cal_sod(gcn_logits_sigmoid_val[cum_add[one]: cum_add[one+1]].tolist(), segments[one])

                    mae2 = self._eval_mae(pre_sod, tar_sod)
                    prec2, recall2 = self._eval_pr(pre_sod, tar_sod, th_num)
                    epoch_test_mae2 += mae2
                    epoch_test_prec2 += prec2
                    epoch_test_recall2 += recall2

                    mae3 = self._eval_mae(pre_sod, labels_sp_val[one])
                    prec3, recall3 = self._eval_pr(pre_sod, labels_sp_val[one], th_num)
                    epoch_test_mae3 += mae3
                    epoch_test_prec3 += prec3
                    epoch_test_recall3 += recall3
                    pass

                # cal 3
                mae4 = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
                prec4, recall4 = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
                epoch_test_mae4 += mae4
                epoch_test_prec4 += prec4
                epoch_test_recall4 += recall4

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}+{:.4f})-{:.4f}({:.4f}+{:.4f}) "
                                "sod-mse={:.4f}({:.4f}) gcn-mse={:.4f}({:.4f}) gcn-final-mse={:.4f}({:.4f})".format(
                        i, len(loader), loss.detach().item(), loss1.detach().item(), loss2.detach().item(),
                        epoch_test_loss/(i+1), epoch_test_loss1/(i+1), epoch_test_loss2/(i+1),
                        mae4, epoch_test_mae4/(i+1), mae, epoch_test_mae/(i+1),
                        epoch_test_mae2/nb_data, epoch_test_mae3/nb_data))
                    pass
                pass
            pass

        # 结果1
        avg_loss, avg_mae = epoch_test_loss / len(loader), epoch_test_mae / len(loader)
        avg_loss1, avg_loss2 = epoch_test_loss1 / len(loader), epoch_test_loss2 / len(loader)
        avg_prec, avg_recall = epoch_test_prec / len(loader), epoch_test_recall / len(loader)
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        # 结果2
        avg_mae2, avg_prec2, avg_recall2 = epoch_test_mae2/nb_data, epoch_test_prec2/nb_data, epoch_test_recall2/nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)
        # 结果3
        avg_mae3, avg_prec3, avg_recall3 = epoch_test_mae3/nb_data, epoch_test_prec3/nb_data, epoch_test_recall3/nb_data
        score3 = (1 + 0.3) * avg_prec3 * avg_recall3 / (0.3 * avg_prec3 + avg_recall3)
        # 结果4
        avg_mae4 = epoch_test_mae4/len(loader)
        avg_prec4, avg_recall4 = epoch_test_prec4/len(loader), epoch_test_recall4/len(loader)
        score4 = (1 + 0.3) * avg_prec4 * avg_recall4 / (0.3 * avg_prec4 + avg_recall4)

        return (avg_loss, avg_loss1, avg_loss2, avg_mae, score.max(),
                avg_mae2, score2.max(), avg_mae3, score3.max(), avg_mae4, score4.max())

    def visual(self, model_file=None, is_train=False, result_path=None):
        if model_file:
            self.load_model(model_file_name=model_file)

        if result_path:
            result_path = Tools.new_dir(os.path.join(result_path, "train" if is_train else "test"))

        loader = self.train_loader if is_train else self.test_loader
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels_sp, labels_sod, batched_graph,
                    batched_pixel_graph, segments, images_small, images_name) in enumerate(loader):
                Tools.print("{}-{}".format(i, len(loader)))

                # Data
                images = images.float().to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                # Run
                gcn2_logits, gcn2_logits_sigmoid, sod_logits, sod_logits_sigmoid = self.model.forward(
                    images, batched_graph, batched_pixel_graph)

                # 可视化
                gcn_labels = batched_graph.y
                sod_labels = labels_sod
                gcn_logits_val = gcn2_logits_sigmoid.cpu().detach().numpy()
                sod_logits_val = sod_logits_sigmoid.cpu().detach().numpy()

                cum_add = np.cumsum([0] + batched_graph.num_sp.tolist())
                for one in range(len(segments)):
                    tar_gcn = self._cal_sod(gcn_labels[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_gcn = self._cal_sod(gcn_logits_val[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_sod = sod_logits_val[one][0]
                    tar_sod = sod_labels[one]

                    im0 = Image.fromarray(np.asarray(images_small[one], dtype=np.uint8))
                    im1 = Image.fromarray(np.asarray(tar_gcn * 255, dtype=np.uint8))
                    im2 = Image.fromarray(np.asarray(pre_gcn * 255, dtype=np.uint8))
                    im3 = Image.fromarray(np.asarray(tar_sod * 255, dtype=np.uint8))
                    im4 = Image.fromarray(np.asarray(pre_sod * 255, dtype=np.uint8))
                    if result_path:
                        image_name = os.path.splitext(os.path.basename(images_name[one]))[0]
                        im0.save(os.path.join(result_path, "{}_{}.png".format(image_name, 0)))
                        im1.save(os.path.join(result_path, "{}_{}.bmp".format(image_name, 1)))
                        im2.save(os.path.join(result_path, "{}_{}.bmp".format(image_name, 2)))
                        im3.save(os.path.join(result_path, "{}_{}.bmp".format(image_name, 3)))
                        im4.save(os.path.join(result_path, "{}_{}.bmp".format(image_name, 4)))
                    else:
                        im0.show()
                        im1.show()
                        im2.show()
                        im3.show()
                        im4.show()
                    pass
                pass
            pass
        pass

    @staticmethod
    def _cal_sod(pre, segment):
        result = np.asarray(segment.copy(), dtype=np.float32)
        for i in range(len(pre)):
            result[segment == i] = pre[i]
            pass
        return result

    def _lr(self, epoch):
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
        pass

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
    def _eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def _eval_pr(y_pred, y, th_num=100):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    """
    2020-07-29 06:24:01 E:25, Train sod-mae-score=0.0446-0.9613 gcn-mae-score=0.0767-0.9089 gcn-final-mse-score=0.0763-0.8889(0.0982/0.8889) loss=0.2476(0.1670+0.0806)
    2020-07-29 06:24:01 E:25, Test  sod-mae-score=0.0724-0.8147 gcn-mae-score=0.0902-0.7123 gcn-final-mse-score=0.0910-0.6760(0.1069/0.6760) loss=0.3891(0.2161+0.1730)
    2020-07-29 14:26:18 E:49, Train sod-mae-score=0.0232-0.9797 gcn-mae-score=0.0529-0.9319 gcn-final-mse-score=0.0522-0.9112(0.0745/0.9112) loss=0.1722(0.1286+0.0436)
    2020-07-29 14:26:18 E:49, Test  sod-mae-score=0.0686-0.8081 gcn-mae-score=0.0908-0.7081 gcn-final-mse-score=0.0915-0.6797(0.1077/0.6797) loss=0.4528(0.2372+0.2155)
    
    2020-07-30 08:05:27 E:33, Train sod-mae-score=0.0279-0.9757 gcn-mae-score=0.0588-0.9266 gcn-final-mse-score=0.0582-0.9067(0.0800/0.9067) loss=0.1899(0.1374+0.0525)
    2020-07-30 08:05:27 E:33, Test  sod-mae-score=0.0679-0.8116 gcn-mae-score=0.0890-0.7165 gcn-final-mse-score=0.0897-0.6880(0.1055/0.6880) loss=0.4470(0.2379+0.2091)
    """

    # _data_root_path = "/media/ubuntu/4T/ALISURE/Data/DUTS"
    _data_root_path = "/mnt/4T/Data/SOD/DUTS"

    # _batch_size = 4 * 8
    _batch_size = 1 * 8
    _image_size_train = 224
    _image_size_test = 256

    _train_print_freq = 100
    _test_print_freq = 100
    _num_workers = 16
    _use_gpu = True

    _gpu_id = "0"
    # _gpu_id = "1"

    _epochs = 50  # Super Param Group 1
    _is_sgd = False
    # _weight_decay = 0.0005
    _weight_decay = 0.0
    # _lr = [[0, 0.001]]
    _lr = [[0, 0.0001], [30, 0.00001]]

    # _epochs = 100  # Super Param Group 1
    # _is_sgd = True
    # _weight_decay = 5e-4
    # _lr = [[0, 0.001], [70, 0.0001], [90, 0.00001]]

    _has_mask = False  # Super Param 3

    _improved = True
    _has_bn = True
    _has_residual = True
    _is_normalize = True

    _sp_size, _down_ratio, _conv_layer_num, _model_name = 4, 4, 20, "GCNNet-C2PC2PC2"

    _name = "E2E2-{}_{}_{}_lr0001_3".format(_model_name, _is_sgd, _has_mask)

    _root_ckpt_dir = "./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/{}".format(_name)
    Tools.print("epochs:{} ckpt:{} batch size:{} image size:{}/{} sp size:{} "
                "down_ratio:{} conv_layer_num:{} workers:{} gpu:{} has_mask:{} "
                "has_residual:{} is_normalize:{} has_bn:{} improved:{} is_sgd:{} weight_decay:{}".format(
        _epochs, _root_ckpt_dir, _batch_size, _image_size_train, _image_size_test, _sp_size, _down_ratio,
        _conv_layer_num, _num_workers, _gpu_id, _has_mask, _has_residual,
        _is_normalize, _has_bn, _improved, _is_sgd, _weight_decay))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, batch_size=_batch_size,
                       image_size_train=_image_size_train, image_size_test=_image_size_test,
                       sp_size=_sp_size, is_sgd=_is_sgd, has_mask=_has_mask, lr=_lr,
                       residual=_has_residual, normalize=_is_normalize, down_ratio=_down_ratio,
                       has_bn=_has_bn, improved=_improved, weight_decay=_weight_decay, conv_layer_num=_conv_layer_num,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.load_model("./ckpt2/dgl/1_PYG_CONV_Fast-ImageNet/0_4_4_20/epoch_14.pkl")
    runner.train(_epochs, start_epoch=0)

    # runner.visual(model_file="./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/E2E-GCNNet-C2PC2PC2_False_False_lr0001/epoch_49.pkl",
    #               is_train=False, result_path="./result/1_PYG_CONV_Fast-SOD_BAS/E2E-GCNNet-C2PC2PC2_False_False_lr0001")
    pass
