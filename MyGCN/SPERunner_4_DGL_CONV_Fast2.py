import os
import cv2
import dgl
import glob
import time
import torch
import skimage
import numpy as np
from PIL import Image
import torch.nn as nn
from skimage import io
import matplotlib.pyplot as plt
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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // 2
        self.data_root_path = data_root_path

        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        img_data = transforms.Compose([transforms.ToTensor()])(np.asarray(img)).unsqueeze(dim=0)

        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data)
        return graph, pixel_graph, img_data, target

    def get_sp_info(self, img):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size_for_sp,
                                          super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = dgl.DGLGraph()
        graph.add_nodes(len(pixel_adj))
        graph.add_edges(sp_adj[0][:, 0], sp_adj[0][:, 1])
        graph.edata['feat'] = torch.from_numpy(sp_adj[1]).unsqueeze(1).float()
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = dgl.DGLGraph()
            small_graph.add_nodes(len(super_pixel[0]))
            small_graph.ndata['data_where'] = torch.from_numpy(super_pixel[0]).long()
            small_graph.add_edges(super_pixel[1][:, 0], super_pixel[1][:, 1])
            small_graph.edata['feat'] = torch.from_numpy(super_pixel[2]).unsqueeze(1).float()
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
        _nodes_num = [graph.number_of_nodes() for graph in graphs]
        _edges_num = [graph.number_of_edges() for graph in graphs]
        nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()
        batched_graph = dgl.batch(graphs)

        # 像素图
        _pixel_graphs = []
        for super_pixel_i, pixel_graph in enumerate(pixel_graphs):
            for now_graph in pixel_graph:
                now_graph.ndata["data_where"][:, 0] = super_pixel_i
                _pixel_graphs.append(now_graph)
            pass
        _nodes_num = [graph.number_of_nodes() for graph in _pixel_graphs]
        _edges_num = [graph.number_of_edges() for graph in _pixel_graphs]
        pixel_nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        pixel_edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()
        batched_pixel_graph = dgl.batch(_pixel_graphs)

        return (images, labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt,
                batched_pixel_graph, pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt)

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


class CONVNet(nn.Module):

    def __init__(self, in_dim=3, hidden_dims=["64", "64", "M", "128", "128"], out_dim=128, has_bn=True):
        super().__init__()
        self.hidden_dims = hidden_dims

        layers = []
        _in_dim = in_dim
        for index, hidden_dim in enumerate(self.hidden_dims):
            if hidden_dim == "M":
                layers.append(nn.MaxPool2d((2, 2)))
            else:
                layers.append(ConvBlock(_in_dim, int(hidden_dim), 1, padding=1, ks=3, has_bn=has_bn))
                _in_dim = int(hidden_dim)
            pass
        layers.append(ConvBlock(_in_dim, out_dim, 1, padding=1, ks=3, has_bn=has_bn))

        self.features = nn.Sequential(*layers)
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


class GCNNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[146, 146], out_dim=146):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 2 <= len(self.hidden_dims) <= 3
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.gcn_1 = GCNLayer(self.hidden_dims[0], self.hidden_dims[1], F.relu,
                              self.dropout, self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 3:
            self.gcn_2 = GCNLayer(self.hidden_dims[1], self.hidden_dims[2], F.relu,
                                  self.dropout, self.graph_norm, self.batch_norm, self.residual)
            pass
        self.gcn_o = GCNLayer(self.hidden_dims[-1], out_dim, F.relu,
                              self.dropout, self.graph_norm, self.batch_norm, self.residual)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        hidden_nodes_feat = self.gcn_1(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 3:
            hidden_nodes_feat = self.gcn_2(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        hidden_nodes_feat = self.gcn_o(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 3 <= len(self.hidden_dims) <= 6
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.gcn_1 = GCNLayer(self.hidden_dims[0], self.hidden_dims[1], F.relu,
                              self.dropout, self.graph_norm, self.batch_norm, self.residual)
        self.gcn_2 = GCNLayer(self.hidden_dims[1], self.hidden_dims[2], F.relu,
                              self.dropout, self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 4:
            self.gcn_3 = GCNLayer(self.hidden_dims[2], self.hidden_dims[3], F.relu,
                                  self.dropout, self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 5:
            self.gcn_4 = GCNLayer(self.hidden_dims[3], self.hidden_dims[4], F.relu,
                                  self.dropout, self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 6:
            self.gcn_5 = GCNLayer(self.hidden_dims[4], self.hidden_dims[5], F.relu,
                                  self.dropout, self.graph_norm, self.batch_norm, self.residual)

        self.gcn_o = GCNLayer(self.hidden_dims[-1], out_dim, F.relu,
                              self.dropout, self.graph_norm, self.batch_norm, self.residual)
        self.readout_mlp = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        hidden_nodes_feat = self.gcn_1(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        hidden_nodes_feat = self.gcn_2(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 4:
            hidden_nodes_feat = self.gcn_3(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 5:
            hidden_nodes_feat = self.gcn_4(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 6:
            hidden_nodes_feat = self.gcn_5(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        hidden_nodes_feat = self.gcn_o(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GraphSageNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[108, 108], out_dim=108):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 2 <= len(self.hidden_dims) <= 3
        self.dropout = 0.0
        self.residual = True
        self.sage_aggregator = "meanpool"

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.graph_sage_1 = GraphSageLayer(self.hidden_dims[0], self.hidden_dims[1], F.relu,
                                           self.dropout, self.sage_aggregator, self.residual)
        if len(self.hidden_dims) >= 3:
            self.graph_sage_2 = GraphSageLayer(self.hidden_dims[1], self.hidden_dims[2], F.relu,
                                               self.dropout, self.sage_aggregator, self.residual)
            pass
        self.graph_sage_o = GraphSageLayer(self.hidden_dims[-1], out_dim, F.relu,
                                           self.dropout, self.sage_aggregator, self.residual)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        hidden_nodes_feat = self.graph_sage_1(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 3:
            hidden_nodes_feat = self.graph_sage_2(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        hidden_nodes_feat = self.graph_sage_o(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GraphSageNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[108, 108, 108, 108], out_dim=108, n_classes=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 3 <= len(self.hidden_dims) <= 6
        self.dropout = 0.0
        self.residual = True
        self.sage_aggregator = "meanpool"

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.graph_sage_1 = GraphSageLayer(self.hidden_dims[0], self.hidden_dims[1], F.relu,
                                           self.dropout, self.sage_aggregator, self.residual)
        self.graph_sage_2 = GraphSageLayer(self.hidden_dims[1], self.hidden_dims[2], F.relu,
                                           self.dropout, self.sage_aggregator, self.residual)
        if len(self.hidden_dims) >= 4:
            self.graph_sage_3 = GraphSageLayer(self.hidden_dims[2], self.hidden_dims[3], F.relu,
                                               self.dropout, self.sage_aggregator, self.residual)
        if len(self.hidden_dims) >= 5:
            self.graph_sage_4 = GraphSageLayer(self.hidden_dims[3], self.hidden_dims[4], F.relu,
                                               self.dropout, self.sage_aggregator, self.residual)
        if len(self.hidden_dims) >= 6:
            self.graph_sage_5 = GraphSageLayer(self.hidden_dims[4], self.hidden_dims[5], F.relu,
                                               self.dropout, self.sage_aggregator, self.residual)

        self.graph_sage_o = GraphSageLayer(self.hidden_dims[-1], out_dim, F.relu,
                                           self.dropout, self.sage_aggregator, self.residual)
        self.readout_mlp = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        hidden_nodes_feat = self.graph_sage_1(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        hidden_nodes_feat = self.graph_sage_2(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 4:
            hidden_nodes_feat = self.graph_sage_3(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 5:
            hidden_nodes_feat = self.graph_sage_4(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        if len(self.hidden_dims) >= 6:
            hidden_nodes_feat = self.graph_sage_5(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        hidden_nodes_feat = self.graph_sage_o(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GatedGCNNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[70, 70], out_dim=70):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 2 <= len(self.hidden_dims) <= 3
        self.in_dim_edge = 1
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dims[0])
        self.gated_gcn_1 = GatedGCNLayer(self.hidden_dims[0], self.hidden_dims[1], self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 3:
            self.gated_gcn_2 = GatedGCNLayer(self.hidden_dims[1], self.hidden_dims[2], self.dropout,
                                             self.graph_norm, self.batch_norm, self.residual)
            pass
        self.gated_gcn_o = GatedGCNLayer(self.hidden_dims[-1], out_dim, self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        h, e = self.gated_gcn_1(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        if len(self.hidden_dims) >= 3:
            h, e = self.gated_gcn_2(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass
        h, e = self.gated_gcn_o(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)

        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GatedGCNNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[70, 70, 70, 70], out_dim=70, n_classes=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        assert 3 <= len(self.hidden_dims) <= 6
        self.in_dim_edge = 1
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dims[0])
        self.gated_gcn_1 = GatedGCNLayer(self.hidden_dims[0], self.hidden_dims[1], self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual)
        self.gated_gcn_2 = GatedGCNLayer(self.hidden_dims[1], self.hidden_dims[2], self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 4:
            self.gated_gcn_3 = GatedGCNLayer(self.hidden_dims[2], self.hidden_dims[3], self.dropout,
                                             self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 5:
            self.gated_gcn_4 = GatedGCNLayer(self.hidden_dims[3], self.hidden_dims[4], self.dropout,
                                             self.graph_norm, self.batch_norm, self.residual)
        if len(self.hidden_dims) >= 6:
            self.gated_gcn_5 = GatedGCNLayer(self.hidden_dims[4], self.hidden_dims[5], self.dropout,
                                             self.graph_norm, self.batch_norm, self.residual)

        self.gated_gcn_o = GatedGCNLayer(self.hidden_dims[-1], out_dim, self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual)
        self.readout_mlp = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        h, e = self.gated_gcn_1(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        h, e = self.gated_gcn_2(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        if len(self.hidden_dims) >= 4:
            h, e = self.gated_gcn_3(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        if len(self.hidden_dims) >= 5:
            h, e = self.gated_gcn_4(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        if len(self.hidden_dims) >= 6:
            h, e = self.gated_gcn_5(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        h, e = self.gated_gcn_o(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)

        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        # self.model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 64, 64, "M", 64, 64], out_dim=64)  # 2, 3
        # self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[146, 146], out_dim=146)  # 2, 3
        # self.model_gnn2 = GCNNet2(in_dim=146, hidden_dims=[146, 146, 146], out_dim=146, n_classes=10)  # 3, 6

        # self.model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 64, 64, "M", 64, 64], out_dim=64)  # 2, 3
        # self.model_gnn1 = GatedGCNNet1(in_dim=64, hidden_dims=[70, 70], out_dim=70)  # 2, 3
        # self.model_gnn2 = GatedGCNNet2(in_dim=70, hidden_dims=[70, 70, 70], out_dim=70, n_classes=10)  # 3, 6

        self.model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 128, 128], out_dim=128)  # 2, 3

        # self.model_gnn1 = GatedGCNNet1(in_dim=128, hidden_dims=[140, 140], out_dim=140)  # 2, 3
        # self.model_gnn2 = GatedGCNNet2(in_dim=140, hidden_dims=[280, 280, 280], out_dim=280, n_classes=10)  # 3, 6

        self.model_gnn1 = GCNNet1(in_dim=128, hidden_dims=[292, 292], out_dim=292)  # 2, 3
        self.model_gnn2 = GCNNet2(in_dim=292, hidden_dims=[584, 584, 584], out_dim=584, n_classes=10)  # 3, 6
        pass

    def forward(self, images, batched_graph, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, pixel_data_where,
                batched_pixel_graph, pixel_edges_feat, pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt):
        # model 1
        conv_feature = self.model_conv(images) if self.model_conv else images

        # model 2
        pixel_nodes_feat = conv_feature[pixel_data_where[:, 0], :, pixel_data_where[:, 1], pixel_data_where[:, 2]]
        batched_pixel_graph.ndata['feat'] = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph, pixel_nodes_feat, pixel_edges_feat,
                                               pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt)

        # model 3
        batched_graph.ndata['feat'] = gcn1_feature
        logits = self.model_gnn2.forward(batched_graph, gcn1_feature, edges_feat,
                                         nodes_num_norm_sqrt, edges_num_norm_sqrt)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10',
                 batch_size=64, image_size=32, sp_size=4, train_print_freq=100, test_print_freq=50,
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path,
                                       is_train=True, image_size=image_size, sp_size=sp_size)
        self.test_dataset = MyDataset(data_root_path=data_root_path,
                                      is_train=False, image_size=image_size, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet().to(self.device)
        # self.lr_s = [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
        # self.lr_s = [[0, 0.1], [40, 0.01], [70, 0.001], [90, 0.0001]]
        self.lr_s = [[0, 0.1], [100, 0.01], [180, 0.001], [250, 0.0001]]
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][0], momentum=0.9, weight_decay=5e-4)
        self.loss_class = nn.CrossEntropyLoss().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
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

    def _train_epoch(self):
        self.model.train()
        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for i, (images, labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt, batched_pixel_graph,
                pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels = labels.long().to(self.device)
            edges_feat = batched_graph.edata['feat'].to(self.device)
            nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
            edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.device)
            pixel_data_where = batched_pixel_graph.ndata["data_where"].to(self.device)
            pixel_edges_feat = batched_pixel_graph.edata['feat'].to(self.device)
            pixel_nodes_num_norm_sqrt = pixel_nodes_num_norm_sqrt.to(self.device)
            pixel_edges_num_norm_sqrt = pixel_edges_num_norm_sqrt.to(self.device)

            # Run
            self.optimizer.zero_grad()
            logits = self.model.forward(images, batched_graph, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt,
                                        pixel_data_where, batched_pixel_graph, pixel_edges_feat,
                                        pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt)
            loss = self.loss_class(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Stat
            nb_data += labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self._accuracy(logits, labels)

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_train_acc/nb_data))
                pass
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self):
        self.model.eval()

        Tools.print()
        epoch_test_loss, epoch_test_acc, nb_data = 0, 0, 0
        with torch.no_grad():
            for i, (images, labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt, batched_pixel_graph,
                    pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                labels = labels.long().to(self.device)
                edges_feat = batched_graph.edata['feat'].to(self.device)
                nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
                edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.device)
                pixel_data_where = batched_pixel_graph.ndata["data_where"].to(self.device)
                pixel_edges_feat = batched_pixel_graph.edata['feat'].to(self.device)
                pixel_nodes_num_norm_sqrt = pixel_nodes_num_norm_sqrt.to(self.device)
                pixel_edges_num_norm_sqrt = pixel_edges_num_norm_sqrt.to(self.device)

                # Run
                logits = self.model.forward(images, batched_graph, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt,
                                            pixel_data_where, batched_pixel_graph, pixel_edges_feat,
                                            pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt)
                loss = self.loss_class(logits, labels)

                # Stat
                nb_data += labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self._accuracy(logits, labels)

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1), loss.detach().item(), epoch_test_acc/nb_data))
                    pass
                pass
            pass

        return epoch_test_loss / (len(self.test_loader) + 1), epoch_test_acc / nb_data

    def _lr(self, epoch):
        # [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
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
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    """
    GCN       Baseline Has Sigmoid             2020-04-08 15:41:33 Epoch: 97, Train: 0.7781/0.6535 Test: 0.7399/0.8137
    
    GCN       251273 3Conv 2GCN1 4GCN2 4spsize 2020-04-18 12:37:39 Epoch: 93, Train: 0.9585/0.1166 Test: 0.8750/0.4803
    GCN       251273 3Conv 2GCN1 4GCN2 6spsize 2020-04-18 10:01:07 Epoch: 97, Train: 0.9502/0.1414 Test: 0.8678/0.4937
    GCN      1187018 3Conv 2GCN1 4GCN2 4spsize 2020-04-18 22:27:57 Epoch: 79, Train: 0.9656/0.0958 Test: 0.8795/0.4865
    SageNet   244387 3Conv 2GCN1 4GCN2 4spsize 2020-04-19 12:38:14 Epoch: 79, Train: 0.9754/0.0685 Test: 0.8867/0.4702
    SageNet  2006218 3Conv 2GCN1 4GCN2 4spsize 2020-04-19 00:34:45 Epoch: 75, Train: 0.9940/0.0175 Test: 0.8988/0.5567
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize 2020-04-20 05:33:01 Epoch: 90, Train: 0.9693/0.0877 Test: 0.8932/0.4230
    GatedGCN 4478410 3Conv 2GCN1 4GCN2 4spsize 2020-04-20 13:23:06 Epoch: 77, Train: 0.9970/0.0092 Test: 0.9072/0.5581
    
    # ConvNet
    GCN       177161 1Conv 2GCN1 4GCN2 4spsize 2020-04-20 05:49:03 Epoch: 98, Train: 0.9040/0.2680 Test: 0.8283/0.5756
    GCN       166335 0Conv 2GCN1 4GCN2 4spsize 2020-04-21 06:03:26 Epoch: 98, Train: 0.4885/1.4157 Test: 0.4744/1.4767
    
    # LR
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize lr 2020-04-21 04:57 Epoch: 83, Train: 0.9360/0.1844 Test: 0.8812/0.3784
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize lr-wd 2020-04-21 09 Epoch: 92, Train: 0.6860/0.8664 Test: 0.6847/0.8777
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize lr-wd-sgd 2020-04-2 Epoch: 93, Train: 0.9395/0.1752 Test: 0.8847/0.3645
    
    # DataAug
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize da 2020-04-22 07:49 Epoch: 97, Train: 0.8852/0.3292 Test: 0.8626/0.4107
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize da2 2020-04-22 12:2 Epoch: 62, Train: 0.8763/0.3528 Test: 0.8596/0.4243
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize da3 2020-04-22 12:2 Epoch: 56, Train: 0.9201/0.2289 Test: 0.8694/0.4134
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize da4 2020-04-22 12:0 Epoch: 54, Train: 0.9003/0.2859 Test: 0.8691/0.3995
    
    # Norm
    GCN       251273 3Conv 2GCN1 4GCN2 4spsize norm 2020-04-23 08: Epoch: 78, Train: 0.9543/0.1294 Test: 0.8742/0.4595
    GatedGCN  239889 3Conv 2GCN1 4GCN2 4spsize norm 2020-04-23 09: Epoch: 81, Train: 0.9621/0.1063 Test: 0.8832/0.4342
    
    
    GCNNet-norm-small 303631 pool              2020-04-24 01:35:35 Epoch: 76, Train: 0.9672/0.0916 Test: 0.8850/0.4574
    GatedGCNNet-norm-small 288871 pool         2020-04-24 09:42:54 Epoch: 92, Train: 0.9777/0.0623 Test: 0.8876/0.5000
    GatedGCNNet-norm-small-sgd 288871 pool     2020-04-24 10:16:10 Epoch: 94, Train: 0.9118/0.2568 Test: 0.8574/0.4659
    GatedGCNNet-norm-small-sgd-lr 288871 pool  2020-04-24 09:01:59 Epoch: 81, Train: 0.9576/0.1246 Test: 0.8980/0.3493
    
    GatedGCNNet-small-sgd-lr-300 288871 1pool      2020-04-25 10:2 Epoch: 184, Train: 0.9687/0.0932 Test: 0.9023/0.3263
    GatedGCNNet-norm-small-sgd-lr-300 2888711 pool 2020-04-25 20:4 Epoch: 269, Train: 0.9879/0.0355 Test: 0.9013/0.4252
    GCNNet-small-sgd-lr-300 303631 1pool           2020-04-25 18:5 Epoch: 256, Train: 0.9875/0.0385 Test: 0.9026/0.4346
    GCNNet-norm-small-sgd-lr-300 303631 1pool      2020-04-25 11:3 Epoch: 193, Train: 0.9754/0.0750 Test: 0.9052/0.3510
    
    GatedGCNNet-small-sgd-lr-300 362983 2pool 1024 2020-04-26 07:2 Epoch: 241, Train: 0.9961/0.0128 Test: 0.9126/0.4303
    GCNNet-small-sgd-lr-300      377743 2pool 1024 2020-04-26 10:1 Epoch: 291, Train: 0.9962/0.0132 Test: 0.9106/0.4261
    GatedGCNNet-small-sgd-lr-300 362983 2pool   64 2020-04-27 10:2 Epoch: 281, Train: 0.9887/0.0347 Test: 0.9130/0.3853
    GCNNet-small-sgd-lr-300      377743 2pool   64 2020-04-27 09:4 Epoch: 273, Train: 0.9902/0.0306 Test: 0.9151/0.3820
    """
    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("GCNNet")
    # _batch_size = 64
    # _image_size = 32
    # _sp_size = 4
    # _train_print_freq = 1
    # _test_print_freq = 1
    # _num_workers = 1
    # _use_gpu = False
    # _gpu_id = "1"

    # _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    _data_root_path = '/home/ubuntu/ALISURE/data/cifar'
    _root_ckpt_dir = "./ckpt2/dgl/4_DGL_CONV/{}-sgd-lr-300".format("GCN")
    _batch_size = 64
    _image_size = 32
    _sp_size = 3
    _epochs = 300
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 8
    _use_gpu = True
    # _gpu_id = "0"
    _gpu_id = "1"

    Tools.print("ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs)

    pass
