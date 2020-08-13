import os
import cv2
import time
import glob
import torch
import skimage
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg13_bn, vgg16_bn
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv


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

    def __init__(self, image_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        # self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
        #                                  sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                         sigma=slic_sigma, max_iter=slic_max_iter)

        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        edge_index, edge_w, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(self.segment == i, selem=skimage.morphology.square(3))

            _adj_dis = []
            for sp_id in np.unique(self.segment[_now_adj]):
                if sp_id != i:
                    edge_index.append([i, sp_id])
                    _adj_dis.append(1/(np.sqrt((self.region_props[i][0][0] - self.region_props[sp_id][0][0]) ** 2 +
                                               (self.region_props[i][0][1] - self.region_props[sp_id][0][1]) ** 2) + 1))
                pass
            edge_w.extend(_adj_dis / np.sum(_adj_dis, axis=0))

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            _dis = _dis <= 2
            pixel_edge_index = np.argwhere(_dis)
            pixel_edge_w = np.ones(len(pixel_edge_index))
            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w])
            pass

        sp_adj = [np.asarray(edge_index), np.asarray(edge_w)]
        return self.segment, sp_adj, pixel_adj

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4, down_ratio=1):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // down_ratio
        self.data_root_path = data_root_path

        # self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4),
        #                                      transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=2),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        # self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=None),
        #                                      transforms.RandomHorizontalFlip()]) if self.is_train else None
        # self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=2),
        #                                      transforms.RandomGrayscale(p=0.2),
        #                                      transforms.RandomHorizontalFlip()]) if self.is_train else None

        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        img_data = transforms.Compose([transforms.ToTensor(), normalize])(np.asarray(img)).unsqueeze(dim=0)

        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data, target)
        return graph, pixel_graph, img_data, target

    def get_sp_info(self, img, target):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size_for_sp,
                                          super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = Data(edge_index=torch.from_numpy(np.transpose(sp_adj[0], axes=(1, 0))),
                     num_nodes=len(pixel_adj), y=torch.tensor([target]),
                     edge_w=torch.from_numpy(sp_adj[1]).unsqueeze(1).float())
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([target]),
                               edge_w=torch.from_numpy(super_pixel[2]).unsqueeze(1).float())
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph

    @staticmethod
    def collate_fn(samples):
        graphs, pixel_graphs, images, labels = map(list, zip(*samples))
        images = torch.cat(images)
        labels = torch.tensor(np.array(labels))

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

        return images, labels, batched_graph, batched_pixel_graph

    pass


class CONVNet(nn.Module):

    def __init__(self, layer_num=6, out_dim=None):  # 6, 13
        super().__init__()
        if out_dim:
            layers = [nn.Conv2d(3, out_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)
        else:
            self.features = vgg13_bn(pretrained=True).features[0: layer_num]
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


class CNNNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128,], has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(nn.Linear(_in_dim, hidden_dim))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_1(hidden_nodes_feat, data.batch)
        return hg

    pass


class CNNNet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128], n_classes=10,
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(nn.Linear(_in_dim, hidden_dim))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_2(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class GCNNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_1(hidden_nodes_feat, data.batch)
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10,
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_2(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.concat = concat

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_1(hidden_nodes_feat, data.batch)
        return hg

    pass


class SAGENet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10,
                 has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_2(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class GATNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, residual=False, concat=False, heads=8):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims[:-2]:
            # assert hidden_dim % heads == 0
            # self.gcn_list.append(GATConv(_in_dim, hidden_dim // heads, heads=heads, concat=self.concat))
            self.gcn_list.append(GATConv(_in_dim, hidden_dim, heads=heads, concat=self.concat))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GATConv(_in_dim, self.hidden_dims[-1], heads=1, concat=self.concat))

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_1(hidden_nodes_feat, data.batch)
        return hg

    pass


class GATNet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10,
                 has_bn=False, residual=False, concat=False, heads=8):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims[:-2]:
            # assert hidden_dim % heads == 0
            # self.gcn_list.append(GATConv(_in_dim, hidden_dim // heads, heads=heads, concat=self.concat))
            self.gcn_list.append(GATConv(_in_dim, hidden_dim, heads=heads, concat=self.concat))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GATConv(_in_dim, self.hidden_dims[-1], heads=1, concat=self.concat))

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_pool_2(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class ResGatedGCN(MessagePassing):

    def __init__(self, in_channels, out_channels, has_bn=True, normalize=False, bias=True, **kwargs):
        super(ResGatedGCN, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bn = has_bn
        self.normalize = normalize

        self.U = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.V = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.A = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.B = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.C = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        self.bn_node_h = nn.BatchNorm1d(self.out_channels) if self.has_bn else None
        self.bn_node_e = nn.BatchNorm1d(self.out_channels) if self.has_bn else None
        self.relu = nn.ReLU()
        self.e = None

        self.reset_parameters()
        pass

    def reset_parameters(self):
        self.U.reset_parameters()
        self.V.reset_parameters()
        self.A.reset_parameters()
        self.B.reset_parameters()
        self.C.reset_parameters()
        if self.has_bn:
            self.bn_node_h.reset_parameters()
            self.bn_node_e.reset_parameters()
        pass

    def forward(self, x, e, edge_index, edge_weight=None, size=None, res_n_id=None):
        Uh = self.U(x)
        Vh = self.V(x)
        Ah = self.A(x)
        Bh = self.B(x)
        self.e = self.C(e)

        h = self.propagate(edge_index, size=size, x=x, Ah=Ah, Bh=Bh,
                           Uh=Uh, Vh=Vh, edge_weight=edge_weight, res_n_id=res_n_id)
        e = self.e
        if self.has_bn:  # batch normalization
            h, e = self.bn_node_h(h), self.bn_node_e(e)
            pass

        h, e = self.relu(h), self.relu(e)
        return h, e

    def message(self, x, Ah_i, Bh_j, Uh_i, Vh_j, edge_index_i, size_i):
        e_ij = Ah_i + Bh_j + self.e
        self.e = e_ij
        sigma_ij = torch.sigmoid(e_ij)

        # 1
        # a = Vh_j * sigma_ij
        # b = scatter_add(sigma_ij, edge_index_i, dim=0, dim_size=size_i)[edge_index_i] + 1e-16

        # 2
        a = scatter_add(Vh_j * sigma_ij, edge_index_i, dim=0, dim_size=size_i)[edge_index_i]
        b = scatter_add(sigma_ij, edge_index_i, dim=0, dim_size=size_i)[edge_index_i] + 1e-16

        return Uh_i + a / b

    def update(self, aggr_out):
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

    pass


class ResGatedGCN1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128, 128, 128], has_bn=False, normalize=False, residual=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.embedding_e = nn.Linear(1, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(ResGatedGCN(_in_dim, hidden_dim, normalize=self.normalize, has_bn=self.has_bn))
            _in_dim = hidden_dim
            pass
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        hidden_edges_feat = self.embedding_e(data.edge_w)
        for gcn in self.gcn_list:
            h_in, e_in = hidden_nodes_feat, hidden_edges_feat
            hidden_nodes_feat, hidden_edges_feat = gcn(h_in, e_in, data.edge_index, data.edge_w)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
                hidden_edges_feat = e_in + hidden_edges_feat
            pass

        hg = global_pool_1(hidden_nodes_feat, data.batch)
        return hg

    pass


class ResGatedGCN2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 n_classes=10, has_bn=False, normalize=False, residual=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.embedding_e = nn.Linear(1, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(ResGatedGCN(_in_dim, hidden_dim, normalize=self.normalize, has_bn=self.has_bn))
            _in_dim = hidden_dim
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        hidden_edges_feat = self.embedding_e(data.edge_w)
        for gcn in self.gcn_list:
            h_in, e_in = hidden_nodes_feat, hidden_edges_feat
            hidden_nodes_feat, hidden_edges_feat = gcn(h_in, e_in, data.edge_index, data.edge_w)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
                hidden_edges_feat = e_in + hidden_edges_feat
            pass

        hg = global_pool_2(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=6, which=0,
                 has_bn=False, normalize=False, residual=False, improved=False, concat=False):
        super().__init__()
        self.model_conv = CONVNet(layer_num=conv_layer_num)  # 6, 13

        if which == 0:
            # 136576
            # self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
            #                           hidden_dims=[128, 128],
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            # self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
            #                           hidden_dims=[128, 128, 128], n_classes=10,
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

            # 153344
            # 165696 2020-07-29 02:51:09 Epoch:147, Train:0.9513-0.9996/0.1499 Test:0.8704-0.9952/0.4132 padding=2
            self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
                                      hidden_dims=[128, 128],
                                      has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
                                      hidden_dims=[128, 128, 128, 128], n_classes=10,
                                      has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

            # 320000 Epoch:00, Train:0.9801-1.0000/0.0714 Test:0.8717-0.9950/0.4560
            # 562496 Epoch:81, Train:0.9750-0.9999/0.0791 Test:0.9087-0.9979/0.3035
            # self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
            #                           hidden_dims=[128, 128],
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            # self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
            #                           hidden_dims=[256, 256, 256, 256], n_classes=10,
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

            # 586240 Epoch:146, Train:0.9888-1.0000/0.0488 Test:0.8712-0.9940/0.4569
            # 828736 Epoch:112, Train:0.9904-1.0000/0.0364 Test:0.9138-0.9964/0.3171
            # self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
            #                           hidden_dims=[128, 128],
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            # self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
            #                           hidden_dims=[256, 256, 512, 512], n_classes=10,
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

            # 170112
            # self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
            #                           hidden_dims=[128, 128],
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            # self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
            #                           hidden_dims=[128, 128, 128, 128, 128], n_classes=10,
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

            # 203648
            # self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[-2].num_features,
            #                           hidden_dims=[128, 128, 128],
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
            # self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1],
            #                           hidden_dims=[128, 128, 128, 128, 128, 128], n_classes=10,
            #                           has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
        elif which == 1:
            self.model_gnn1 = SAGENet1(in_dim=self.model_conv.features[-2].num_features,
                                       hidden_dims=[128, 128],
                                       has_bn=has_bn, normalize=normalize, residual=residual, concat=concat)
            self.model_gnn2 = SAGENet2(in_dim=self.model_gnn1.hidden_dims[-1],
                                       hidden_dims=[128, 128, 128, 128], n_classes=10,
                                       has_bn=has_bn, normalize=normalize, residual=residual, concat=concat)
        elif which == 2:
            self.model_gnn1 = GATNet1(in_dim=self.model_conv.features[-2].num_features,
                                      hidden_dims=[128, 128],
                                      has_bn=has_bn, residual=residual, concat=concat, heads=8)
            self.model_gnn2 = GATNet2(in_dim=self.model_gnn1.hidden_dims[-1],
                                      hidden_dims=[128, 128, 128, 128], n_classes=10,
                                      has_bn=has_bn, residual=residual, concat=concat, heads=8)
        elif which == 3:
            self.model_gnn1 = ResGatedGCN1(in_dim=self.model_conv.features[-2].num_features, hidden_dims=[128, 128],
                                           has_bn=has_bn, normalize=normalize, residual=residual)
            self.model_gnn2 = ResGatedGCN2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[128, 128, 128, 128],
                                           has_bn=has_bn, normalize=normalize, residual=residual, n_classes=10)
        elif which == 4:
            self.model_gnn1 = CNNNet1(in_dim=self.model_conv.features[-2].num_features, hidden_dims=[128, 128],
                                      has_bn=has_bn, normalize=normalize, residual=residual)
            self.model_gnn2 = CNNNet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[128, 128, 128, 128],
                                      has_bn=has_bn, normalize=normalize, residual=residual, n_classes=10)
        else:
            assert which == -1
        pass

    def forward(self, images, batched_graph, batched_pixel_graph):
        # model 1
        conv_feature = self.model_conv(images)

        # model 2
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = conv_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph)

        # model 3
        batched_graph.x = gcn1_feature
        logits = self.model_gnn2.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', down_ratio=1, concat=False, which=0,
                 batch_size=64, image_size=32, sp_size=4, train_print_freq=100, test_print_freq=50, lr=None,
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1", conv_layer_num=6,
                 has_bn=True, normalize=True, residual=False, improved=False, weight_decay=0.0, is_sgd=False):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                       is_train=True, image_size=image_size, sp_size=sp_size)
        self.test_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                      is_train=False, image_size=image_size, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=conv_layer_num, which=which, has_bn=has_bn, normalize=normalize,
                              residual=residual, improved=improved, concat=concat).to(self.device)

        if is_sgd:
            self.lr_s = [[0, 0.1], [50, 0.01], [100, 0.001]] if lr is None else lr
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][1],
                                             momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [50, 0.0002], [75, 0.00004]] if lr is None else lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=weight_decay)

        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))

        self.loss_class = nn.CrossEntropyLoss().to(self.device)
        pass

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

            epoch_loss, epoch_train_acc, epoch_train_acc_k = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc, epoch_test_acc_k = self.test()

            Tools.print('Epoch:{:02d}, Train:{:.4f}-{:.4f}/{:.4f} Test:{:.4f}-{:.4f}/{:.4f}'.format(
                epoch, epoch_train_acc, epoch_train_acc_k,
                epoch_loss, epoch_test_acc, epoch_test_acc_k, epoch_test_loss))
            pass
        pass

    def _train_epoch(self):
        self.model.train()

        epoch_loss, epoch_train_acc, epoch_train_acc_k, nb_data = 0, 0, 0, 0
        for i, (images, labels, batched_graph, batched_pixel_graph) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels = labels.long().to(self.device)

            batched_graph.batch = batched_graph.batch.to(self.device)
            batched_graph.edge_w = batched_graph.edge_w.to(self.device)
            batched_graph.edge_index = batched_graph.edge_index.to(self.device)

            batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
            batched_pixel_graph.edge_w = batched_pixel_graph.edge_w.to(self.device)
            batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
            batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

            # Run
            self.optimizer.zero_grad()
            logits = self.model.forward(images, batched_graph, batched_pixel_graph)
            loss = self.loss_class(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Stat
            nb_data += labels.size(0)
            epoch_loss += loss.detach().item()
            top_1, top_k = self._accuracy_top_k(logits, labels)
            epoch_train_acc += top_1
            epoch_train_acc_k += top_k

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{}-{} loss={:.4f}/{:.4f} acc={:.4f} acc5={:.4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1),
                    loss.detach().item(), epoch_train_acc/nb_data, epoch_train_acc_k/nb_data))
                pass
            pass
        return epoch_loss/(len(self.train_loader)+1), epoch_train_acc/nb_data, epoch_train_acc_k/nb_data

    def test(self):
        self.model.eval()

        Tools.print()
        epoch_test_loss, epoch_test_acc, epoch_test_acc_k, nb_data = 0, 0, 0, 0
        with torch.no_grad():
            for i, (images, labels, batched_graph, batched_pixel_graph) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                labels = labels.long().to(self.device)

                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_w = batched_graph.edge_w.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_w = batched_pixel_graph.edge_w.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                # Run
                logits = self.model.forward(images, batched_graph, batched_pixel_graph)
                loss = self.loss_class(logits, labels)

                # Stat
                nb_data += labels.size(0)
                epoch_test_loss += loss.detach().item()
                top_1, top_k = self._accuracy_top_k(logits, labels)
                epoch_test_acc += top_1
                epoch_test_acc_k += top_k

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{}-{} loss={:.4f}/{:.4f} acc={:.4f} acc5={:.4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1),
                        loss.detach().item(), epoch_test_acc/nb_data, epoch_test_acc_k/nb_data))
                    pass
                pass
            pass

        return epoch_test_loss/(len(self.test_loader)+1), epoch_test_acc/nb_data, epoch_test_acc_k/nb_data

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
    def _accuracy(scores, targets):
        return (scores.detach().argmax(dim=1) == targets).float().sum().item()

    @staticmethod
    def _accuracy_top_k(scores, targets, top_k=5):
        top_k_index = scores.detach().topk(top_k)[1]
        top_k = sum([int(a) in b for a, b in zip(targets, top_k_index)])
        top_1 = sum([int(a) == int(b[0]) for a, b in zip(targets, top_k_index)])
        return top_1, top_k

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


"""
GCNNet
SGD  136576 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:121, Train:0.8996-0.9984/0.2875 Test:0.8537-0.9955/0.4331
SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:119, Train:0.8947-0.9978/0.3061 Test:0.8524-0.9936/0.4492
SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:110, Train:0.9120-0.9986/0.2591 Test:0.8645-0.9953/0.4043 padding=2
SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:102, Train:0.9283-0.9990/0.2187 Test:0.8549-0.9951/0.4184 padding=0
SGD  170112 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:125, Train:0.8929-0.9981/0.3061 Test:0.8525-0.9944/0.4434
SGD  203648 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:118, Train:0.8743-0.9970/0.3626 Test:0.8347-0.9935/0.5036
SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:116, Train:0.8399-0.9942/0.4581 Test:0.8265-0.9922/0.5058 padding=2 color
Adam 153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:False weight_decay:0.0    Epoch: 92, Train:0.8865-0.9971/0.3170 Test:0.8505-0.9937/0.4587 padding=2 color
Adam 153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:False weight_decay:0.0    Epoch: 79, Train:0.9354-0.9995/0.1828 Test:0.8559-0.9953/0.5030
SGD  395840 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:116, Train:0.9468-0.9991/0.1577 Test:0.8927-0.9959/0.3575

SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:120, Train:0.9121-0.9989/0.2533 Test:0.8575-0.9933/0.4349 padding=2
SGD  153344 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:124, Train:0.9269-0.9992/0.2153 Test:0.8630-0.9954/0.4132 padding=2
SGD  395840 has_residual:True  is_normalize:True  has_bn:True improved:True is_sgd:True  weight_decay:0.0005 Epoch:113, Train:0.9650-0.9998/0.1059 Test:0.9023-0.9968/0.3181 padding=2


SAGENet
SGD  153344 has_residual:True  is_normalize:True  has_bn:True concat:False  is_sgd:True  weight_decay:0.0005 Epoch:123, Train:0.9106-0.9985/0.2586 Test:0.8570-0.9953/0.4469
SGD  243456 has_residual:True  is_normalize:True  has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:118, Train:0.9318-0.9994/0.1975 Test:0.8744-0.9957/0.4026
SGD  243456 has_residual:False is_normalize:True  has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:113, Train:0.9229-0.9990/0.2216 Test:0.8704-0.9956/0.4058
SGD  243456 has_residual:True  is_normalize:False has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:118, Train:0.9321-0.9992/0.2005 Test:0.8710-0.9960/0.4015
Adam 153344 has_residual:True  is_normalize:True  has_bn:True concat:True   is_sgd:False weight_decay:0.0    Epoch: 80, Train:0.9480-0.9998/0.1458 Test:0.8543-0.9943/0.5667
Adam 243456 has_residual:True  is_normalize:True  has_bn:True concat:True   is_sgd:False weight_decay:0.0    Epoch: 77, Train:0.9672-0.9999/0.0933 Test:0.8706-0.9962/0.5388

SGD  395840 has_residual:True  is_normalize:True  has_bn:True concat:False  is_sgd:True  weight_decay:0.0005 Epoch:108, Train:0.9444-0.9993/0.1604 Test:0.8963-0.9964/0.3470
SGD  494144 has_residual:False is_normalize:True  has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:107, Train:0.9442-0.9992/0.1621 Test:0.8924-0.9962/0.3286
SGD  494144 has_residual:True  is_normalize:True  has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:113, Train:0.9563-0.9994/0.1294 Test:0.8995-0.9960/0.3355

GAT
SGD  121344 has_residual:True  is_normalize:True  has_bn:True concat:True   is_sgd:True  weight_decay:0.0005 Epoch:146, Train:0.8919-0.9979/0.3092 Test:0.8510-0.9944/0.4449
SGD  354304 has_residual:True  is_normalize:True  has_bn:True concat:False  is_sgd:True  weight_decay:0.0005 Epoch:113, Train:0.9232-0.9988/0.2235 Test:0.8615-0.9951/0.4277

ResGatedGCN
SGD  518784 has_residual:True  is_normalize:True  has_bn:True concat:False  is_sgd:True  weight_decay:0.0005 Epoch:121, Train:0.9056-0.9978/0.2796 Test:0.8553-0.9947/0.4318
SGD  518784 has_residual:True  is_normalize:True  has_bn:True concat:False  is_sgd:True  weight_decay:0.0005 Epoch:113, Train:0.9323-0.9994/0.1950 Test:0.8796-0.9967/0.3696

Padding = 2, GCN 维度全一样时比较好

CNN
SGD  4_True_4_1_6_mean_mean_pool 165696 Epoch:143, Train:0.9124-0.9985/0.2658 Test:0.8415-0.9933/0.4722
SGD 4_True_2_2_13_mean_mean_pool 395840 Epoch:138, Train:0.9439-0.9991/0.1725 Test:0.8838-0.9950/0.3714
"""


if __name__ == '__main__':
    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    # _data_root_path = '/home/ubuntu/ALISURE/data/cifar'
    _batch_size = 64
    _image_size = 32
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 20
    _use_gpu = True

    # _which = 0  # GCN
    # _which = 1  # SAGE
    # _which = 2  # GAT
    # _which = 3  # ResGatedGCN
    _which = 4  # CNN

    _gpu_id = "0"
    # _gpu_id = "1"

    # _is_sgd = False
    _is_sgd = True

    # _sp_size, _down_ratio, _conv_layer_num = 4, 1, 6
    _sp_size, _down_ratio, _conv_layer_num = 2, 2, 13

    global_pool_1, global_pool_2, pool_name = global_mean_pool, global_mean_pool, "mean_mean_pool"
    # global_pool_1, global_pool_2, pool_name = global_max_pool, global_max_pool, "max_max_pool"
    # global_pool_1, global_pool_2, pool_name = global_mean_pool, global_max_pool, "mean_max_pool"

    if _which == 0:
        _improved, _has_bn, _has_residual, _is_normalize = True, True, True, True
        _concat = False  # No use
    elif _which == 1:
        _concat, _has_bn, _has_residual, _is_normalize = True, True, True, True
        _improved = True  # No use
    elif _which == 2:
        _concat, _has_bn, _has_residual = False, True, True
        _is_normalize, _improved = True, True  # No use
    elif _which == 3:
        _has_bn, _has_residual, _is_normalize = True, True, False
        _concat, _improved = False, True  # No use
    elif _which == 4:
        _has_bn, _has_residual = True, True
        _is_normalize, _concat, _improved = False, False, True  # No use
    else:
        raise Exception(".......")

    if _is_sgd:
        _epochs, _weight_decay, _lr = 150, 5e-4, [[0, 0.1], [50, 0.01], [100, 0.001], [130, 0.0001]]
        # _lr = [[0, 0.01], [50, 0.001], [100, 0.0001]]
    else:
        _epochs, _weight_decay, _lr = 100, 0.0, [[0, 0.001], [50, 0.0002], [75, 0.00004]]
        pass

    _root_ckpt_dir = "./ckpt2/dgl/1_PYG_CONV_Fast_CIFAR10/{}_{}_{}_{}_{}_{}".format(
        _which, _is_sgd, _sp_size,  _down_ratio, _conv_layer_num, pool_name)
    Tools.print("epochs:{} ckpt:{} batch size:{} image size:{} sp size:{} down_ratio:{} "
                "conv_layer_num:{} workers:{} gpu:{} has_residual:{} is_normalize:{} "
                "has_bn:{} improved:{} concat:{} is_sgd:{} weight_decay:{} pool_name:{}".format(
        _epochs, _root_ckpt_dir, _batch_size, _image_size, _sp_size, _down_ratio, _conv_layer_num, _num_workers,
        _gpu_id, _has_residual, _is_normalize, _has_bn, _improved, _concat, _is_sgd, _weight_decay, pool_name))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, concat=_concat, which=_which,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       residual=_has_residual, normalize=_is_normalize, down_ratio=_down_ratio, lr=_lr,
                       has_bn=_has_bn, improved=_improved, weight_decay=_weight_decay, conv_layer_num=_conv_layer_num,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs)

    pass
