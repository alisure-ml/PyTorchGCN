import os
import cv2
import dgl
import glob
import time
import torch
import shutil
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

    def __init__(self, image_data, ds_image_size=28, super_pixel_size=4, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                         sigma=slic_sigma, max_iter=slic_max_iter, multichannel=False)
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

    def __init__(self, data_root_path='D:\\data\\MNIST', is_train=True, image_size=28, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size
        self.data_root_path = data_root_path

        _transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4)]) if self.is_train else None

        self.data_set = datasets.MNIST(root=self.data_root_path, train=self.is_train, transform=_transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        img_data = transforms.Compose([transforms.ToTensor()])(
            np.expand_dims(np.asarray(img), axis=-1)).unsqueeze(dim=0)

        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data)
        return graph, pixel_graph, img_data, target

    def get_sp_info(self, img):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(img, ds_image_size=self.image_size_for_sp, super_pixel_size=self.sp_size)
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

    def __init__(self, in_dim=3, hidden_dims=["64", "64", "M", "128", "128", "M"], out_dim=128, has_bn=True):
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

    def __init__(self, in_dim=128, hidden_dims=[146, 146], out_dim=146):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNLayer(_in_dim, hidden_dim, F.relu,
                                          self.dropout, self.graph_norm, self.batch_norm, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNLayer(self.hidden_dims[-1], out_dim, F.relu,
                                      self.dropout, self.graph_norm, self.batch_norm, self.residual))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[146, 146, 146], out_dim=146, n_classes=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNLayer(_in_dim, hidden_dim, F.relu,
                                          self.dropout, self.graph_norm, self.batch_norm, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNLayer(self.hidden_dims[-1], out_dim, F.relu,
                                      self.dropout, self.graph_norm, self.batch_norm, self.residual))
        self.readout_mlp = MLPReadout(out_dim, n_classes, L=1)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
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

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GraphSageLayer(_in_dim, hidden_dim, F.relu, self.dropout,
                                                self.dropout, self.sage_aggregator, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GraphSageLayer(self.hidden_dims[-1], out_dim, F.relu, self.dropout,
                                            self.dropout, self.sage_aggregator, self.residual))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass

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

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GraphSageLayer(_in_dim, hidden_dim, F.relu, self.dropout,
                                                self.dropout, self.sage_aggregator, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GraphSageLayer(self.hidden_dims[-1], out_dim, F.relu, self.dropout,
                                            self.dropout, self.sage_aggregator, self.residual))

        self.readout_mlp = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)

        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GatedGCNNet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[70, 70], out_dim=70):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_dim_edge = 1
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dims[0])

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GatedGCNLayer(_in_dim, hidden_dim, self.dropout,
                                               self.graph_norm, self.batch_norm, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GatedGCNLayer(self.hidden_dims[-1], out_dim, self.dropout,
                                           self.graph_norm, self.batch_norm, self.residual))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        for gcn in self.gcn_list:
            h, e = gcn(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GatedGCNNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[70, 70, 70, 70], out_dim=70, n_classes=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_dim_edge = 1
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dims[0])

        _in_dim = self.hidden_dims[0]
        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GatedGCNLayer(_in_dim, hidden_dim, self.dropout,
                                               self.graph_norm, self.batch_norm, self.residual))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GatedGCNLayer(self.hidden_dims[-1], out_dim, self.dropout,
                                           self.graph_norm, self.batch_norm, self.residual))

        self.readout_mlp = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        for gcn in self.gcn_list:
            h, e = gcn(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        # self.model_conv = CONVNet(in_dim=1, hidden_dims=["64"], out_dim=64)
        # self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[128, 128], out_dim=128)
        # self.model_gnn2 = GCNNet2(in_dim=128, hidden_dims=[128, 128, 128], out_dim=128, n_classes=10)

        # self.model_conv = CONVNet(in_dim=1, hidden_dims=[], out_dim=32)
        # self.model_gnn1 = GCNNet1(in_dim=32, hidden_dims=[32, 32], out_dim=32)
        # self.model_gnn2 = GCNNet2(in_dim=32, hidden_dims=[64, 64, 64], out_dim=64, n_classes=10)

        # self.model_gnn1 = GCNNet1(in_dim=1, hidden_dims=[32, 32], out_dim=32)
        # self.model_gnn2 = GCNNet2(in_dim=32, hidden_dims=[64, 64, 64], out_dim=64, n_classes=10)

        # self.model_gnn1 = GCNNet1(in_dim=1, hidden_dims=[32, 32], out_dim=64)
        # self.model_gnn2 = GCNNet2(in_dim=64, hidden_dims=[64, 64, 128], out_dim=128, n_classes=10)

        # self.model_conv = CONVNet(in_dim=1, hidden_dims=["64"], out_dim=64)
        # self.model_gnn1 = GraphSageNet1(in_dim=64, hidden_dims=[128, 128], out_dim=128)
        # self.model_gnn2 = GraphSageNet2(in_dim=128, hidden_dims=[128, 128, 128], out_dim=128, n_classes=10)

        # self.model_conv = CONVNet(in_dim=1, hidden_dims=[], out_dim=32)
        # self.model_gnn1 = GraphSageNet1(in_dim=32, hidden_dims=[32, 32], out_dim=32)
        # self.model_gnn2 = GraphSageNet2(in_dim=32, hidden_dims=[64, 64, 64], out_dim=64, n_classes=10)

        # self.model_gnn1 = GraphSageNet1(in_dim=1, hidden_dims=[32, 32], out_dim=32)
        # self.model_gnn2 = GraphSageNet2(in_dim=32, hidden_dims=[64, 64, 64], out_dim=64, n_classes=10)

        # self.model_gnn1 = GraphSageNet1(in_dim=1, hidden_dims=[32, 32], out_dim=64)
        # self.model_gnn2 = GraphSageNet2(in_dim=64, hidden_dims=[64, 64, 128], out_dim=128, n_classes=10)

        # self.model_conv = CONVNet(in_dim=1, hidden_dims=["64"], out_dim=64)
        # self.model_gnn1 = GatedGCNNet1(in_dim=64, hidden_dims=[70, 70], out_dim=70)
        # self.model_gnn2 = GatedGCNNet2(in_dim=70, hidden_dims=[70, 70, 70], out_dim=70, n_classes=10)

        # self.model_conv = CONVNet(in_dim=1, hidden_dims=[], out_dim=32)
        # self.model_gnn1 = GatedGCNNet1(in_dim=32, hidden_dims=[16, 16], out_dim=16)
        # self.model_gnn2 = GatedGCNNet2(in_dim=16, hidden_dims=[32, 32, 32], out_dim=32, n_classes=10)

        # self.model_gnn1 = GatedGCNNet1(in_dim=1, hidden_dims=[16, 16], out_dim=16)
        # self.model_gnn2 = GatedGCNNet2(in_dim=16, hidden_dims=[32, 32, 32], out_dim=32, n_classes=10)

        self.model_gnn1 = GatedGCNNet1(in_dim=1, hidden_dims=[16, 16], out_dim=35)
        self.model_gnn2 = GatedGCNNet2(in_dim=35, hidden_dims=[35, 35, 70], out_dim=70, n_classes=10)
        pass

    def forward(self, images, batched_graph, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, pixel_data_where,
                batched_pixel_graph, pixel_edges_feat, pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt):
        # model 1
        # conv_feature = self.model_conv(images)
        conv_feature = images

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
                 batch_size=64, image_size=28, sp_size=4, train_print_freq=100, test_print_freq=50,
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
        self.lr_s = [[0, 0.001], [10, 0.003], [20, 0.0001]]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)
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
            epoch_loss, epoch_train_acc, epoch_train_acc_k = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc, epoch_test_acc_k = self.test()

            Tools.print('Epoch:{:02d},lr={:.4f},Train:{:.4f}-{:.4f}/{:.4f} Test:{:.4f}-{:.4f}/{:.4f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'],
                epoch_train_acc, epoch_train_acc_k, epoch_loss, epoch_test_acc, epoch_test_acc_k, epoch_test_loss))
            pass
        pass

    def _train_epoch(self):
        self.model.train()
        epoch_loss, epoch_train_acc, epoch_train_acc_k, nb_data = 0, 0, 0, 0
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


if __name__ == '__main__':
    """
    GCNNet 239242 2020-05-04 05:41:49 Epoch:27,lr=0.0001,Train:0.9961-1.0000/0.0121 Test:0.9949-1.0000/0.0168
    GCNNet  36170 2020-05-04 04:01:07 Epoch:22,lr=0.0001,Train:0.9919-1.0000/0.0261 Test:0.9932-1.0000/0.0197
    GCNNet  34794 2020-05-04 06:05:31 Epoch:28,lr=0.0001,Train:0.9526-0.9988/0.1459 Test:0.9652-0.9990/0.1085
    """
    # _data_root_path = 'D:\\data\\MNIST'
    # _root_ckpt_dir = "ckpt3\\dgl\\my\\{}".format("GCNNet")
    # _batch_size = 64
    # _image_size = 28
    # _sp_size = 4
    # _train_print_freq = 1
    # _test_print_freq = 1
    # _num_workers = 1
    # _use_gpu = False
    # _gpu_id = "1"

    _data_root_path = '/home/ubuntu/ALISURE/data/mnist'
    _root_ckpt_dir = "./ckpt2/dgl/4_DGL_CONV-mnist/{}".format("GCNNet")
    _batch_size = 64
    _image_size = 28
    _sp_size = 4
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 4
    _use_gpu = True
    # _gpu_id = "0"
    _gpu_id = "1"

    Tools.print("ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(30)

    pass
