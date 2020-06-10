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
from torchvision.models import vgg13_bn, vgg16_bn


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

    def __init__(self, in_dim, hidden_dims, no_conv_dim=None):
        super().__init__()

        layers = []
        for index, hidden_dim in enumerate(hidden_dims):
            if hidden_dim == "M":
                layers.append(nn.MaxPool2d((2, 2)))
            else:
                layers.append(ConvBlock(in_dim, int(hidden_dim), 1, padding=1, ks=3, has_bn=True))
                in_dim = int(hidden_dim)
            pass

        if no_conv_dim:
            layers = [nn.Conv2d(3, no_conv_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


def readout_fn(readout, graphs, feat):
    if readout == "mean":
        return dgl.mean_nodes(graphs, feat)
    if readout == "max":
        return dgl.max_nodes(graphs, feat)
    return dgl.mean_nodes(graphs, feat)


class GCNNet1(nn.Module):

    def __init__(self, in_dim, hidden_dims, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNLayer(_in_dim, hidden_dim, F.relu, 0.0, True, True, True))
            _in_dim = hidden_dim
            pass

        Tools.print("GCNNet1 #GNN1={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = nodes_feat
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = readout_fn(self.readout, graphs, 'h')
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim, hidden_dims, n_classes=10, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GCNLayer(_in_dim, hidden_dim, F.relu, 0.0, True, True, True))
            _in_dim = hidden_dim
            pass
        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)

        Tools.print("GCNNet2 #GNN2={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = readout_fn(self.readout, graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GraphSageNet1(nn.Module):

    def __init__(self, in_dim, hidden_dims, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GraphSageLayer(_in_dim, hidden_dim, F.relu, 0.0, "meanpool", True))
            _in_dim = hidden_dim
            pass

        Tools.print("GraphSageNet1 #GNN1={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = nodes_feat
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = readout_fn(self.readout, graphs, 'h')
        return hg

    pass


class GraphSageNet2(nn.Module):

    def __init__(self, in_dim, hidden_dims, n_classes=10, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GraphSageLayer(_in_dim, hidden_dim, F.relu, 0.0, "meanpool", True))
            _in_dim = hidden_dim
            pass
        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)

        Tools.print("GraphSageNet2 #GNN2={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass

        graphs.ndata['h'] = hidden_nodes_feat
        hg = readout_fn(self.readout, graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GatedGCNNet1(nn.Module):

    def __init__(self, in_dim, hidden_dims, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.in_dim_edge = 1
        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.embedding_e = nn.Linear(self.in_dim_edge, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GatedGCNLayer(_in_dim, hidden_dim, 0.0, True, True, True))
            _in_dim = hidden_dim
            pass

        Tools.print("GatedGCNNet1 #GNN1={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        for gcn in self.gcn_list:
            h, e = gcn(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = readout_fn(self.readout, graphs, 'h')
        return hg

    pass


class GatedGCNNet2(nn.Module):

    def __init__(self, in_dim, hidden_dims, n_classes=200, readout="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.readout = readout

        self.in_dim_edge = 1
        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.embedding_e = nn.Linear(self.in_dim_edge, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = self.in_dim
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GatedGCNLayer(_in_dim, hidden_dim, 0.0, True, True, True))
            _in_dim = hidden_dim
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], n_classes, bias=False)

        Tools.print("GatedGCNNet2 #GNN2={} in_dim={} hidden_dims={} readout={}".format(
            len(self.hidden_dims), self.in_dim, self.hidden_dims, self.readout))
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        for gcn in self.gcn_list:
            h, e = gcn(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = readout_fn(self.readout, graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self, model_conv=None, model_gnn1=None, model_gnn2=None):
        super().__init__()
        if model_gnn1 and model_gnn2:
            self.model_conv = model_conv
            self.model_gnn1 = model_gnn1
            self.model_gnn2 = model_gnn2
        else:
            ###############
            # 110464 2020-06-03 03:20:43 Epoch: 79, Train: 0.6453/1.0056 Test: 0.5741/1.2538
            # self.model_conv = CONVNet(out_dim=64)
            # self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 149184
            # self.model_conv = CONVNet(layer_num=6)
            # self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 379328 2020-06-05 09:11:32 Epoch: 81, Train: 0.9828/0.0490 Test: 0.8974/0.5142
            # self.model_conv = CONVNet(layer_num=13)
            # self.model_gnn1 = GCNNet1(in_dim=128, hidden_dims=[128, 128])
            # self.model_gnn2 = GCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            ###############
            # 200576 2020-06-02 08:52:37 Epoch: 80, Train: 0.8126/0.5311 Test: 0.7068/0.9594
            # self.model_conv = CONVNet(out_dim=64)
            # self.model_gnn1 = GraphSageNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GraphSageNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 239296 2020-06-05 03:08:02 Epoch: 50, Train: 0.9476/0.1473 Test: 0.8723/0.4600
            # self.model_conv = CONVNet(layer_num=6)
            # self.model_gnn1 = GraphSageNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GraphSageNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 477632 2020-06-01 00:47:31 Epoch: 186, Train: 0.9864/0.0446 Test: 0.9080/0.3570
            # 477632 2020-06-05 08:36:41 Epoch: 80, Train: 0.9902/0.0284 Test: 0.9005/0.5342
            # self.model_conv = CONVNet(layer_num=13)
            # self.model_gnn1 = GraphSageNet1(in_dim=128, hidden_dims=[128, 128])
            # self.model_gnn2 = GraphSageNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            ###############
            # 480064 2020-06-03 00:08:59 Epoch: 58, Train: 0.8852/0.3230 Test: 0.7734/0.7938
            # self.model_conv = CONVNet(out_dim=64)
            # self.model_gnn1 = GatedGCNNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GatedGCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 528170 2020-06-04 06:16:41 Epoch: 79, Train: 0.9877/0.0363 Test: 0.8916/0.5201
            # self.model_conv = CONVNet(layer_num=6)
            # self.model_gnn1 = GatedGCNNet1(in_dim=64, hidden_dims=[128, 128])
            # self.model_gnn2 = GatedGCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)

            # 794176 2020-06-05 08:14:54 Epoch: 86, Train: 0.9937/0.0197 Test: 0.9136/0.4381
            # self.model_conv = CONVNet(layer_num=13)
            # self.model_gnn1 = GatedGCNNet1(in_dim=128, hidden_dims=[128, 128])
            # self.model_gnn2 = GatedGCNNet2(in_dim=128, hidden_dims=[128, 128, 128, 128], n_classes=10)
            pass

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

    def __init__(self, model_conv=None, model_gnn1=None, model_gnn2=None, use_gpu=True, gpu_id="1"):
        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)

        if model_gnn1 and model_gnn2:
            self.model = MyGCNNet(model_conv=model_conv, model_gnn1=model_gnn1, model_gnn2=model_gnn2).to(self.device)
        else:
            self.model = MyGCNNet().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        pass

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    """
    ####################################################################
    1
    ####################################################################
    2020-06-06 19:40:47 #Conv=13 pretrained=False
    2020-06-06 19:40:47 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-06 19:40:47 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-06 19:40:47 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-06 19:40:47 sp-num:64 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:27:46 Total param: 379328
    2020-06-07 20:44:39 Epoch: 109, Train: 0.9887/0.0395 Test: 0.9128/0.3102
    
    2020-06-06 19:59:23 #Conv=13 pretrained=False
    2020-06-06 19:59:23 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-06 19:59:23 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-06 19:59:23 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-06 19:59:23 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:27:46 Total param: 379328
    2020-06-07 09:36:02 Epoch: 61, Train: 0.9697/0.0932 Test: 0.9030/0.3226
    
    2020-06-06 19:59:47 #Conv=13 pretrained=False
    2020-06-06 19:59:47 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-06 19:59:47 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-06 19:59:47 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-06 19:59:47 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:27:46 Total param: 379328
    2020-06-07 08:52:39 Epoch: 56, Train: 0.9660/0.1044 Test: 0.9117/0.2753
    
    2020-06-06 21:15:19 #Conv=13 pretrained=False
    2020-06-06 21:15:19 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-06 21:15:19 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-06 21:15:19 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-06 21:15:19 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:27:46 Total param: 379328
    2020-06-07 12:27:52 Epoch: 67, Train: 0.9736/0.0824 Test: 0.9070/0.3284
    
    
    
    ####################################################################
    2
    ####################################################################
    2020-06-07 21:25:50 #Conv=13 pretrained=False
    2020-06-07 21:25:50 GraphSageNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-07 21:25:50 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-07 21:25:50 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-07 21:25:50 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:25:55 Total param: 477632
    2020-06-08 13:37:12 Epoch: 70, Train: 0.9755/0.0745 Test: 0.9124/0.2953
    
    2020-06-07 21:25:58 #Conv=13 pretrained=False
    2020-06-07 21:25:58 GraphSageNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-07 21:25:58 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-07 21:25:58 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-07 21:25:58 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:26:03 Total param: 477632
    2020-06-08 23:36:12 Epoch: 113, Train: 0.9919/0.0301 Test: 0.9056/0.3902
    
    2020-06-07 21:26:27 #Conv=13 pretrained=False
    2020-06-07 21:26:27 GraphSageNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-07 21:26:27 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-07 21:26:27 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-07 21:26:27 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:26:32 Total param: 477632
    2020-06-08 21:01:27 Epoch: 102, Train: 0.9894/0.0354 Test: 0.9137/0.3240
    
    2020-06-07 21:26:32 #Conv=13 pretrained=False
    2020-06-07 21:26:32 GraphSageNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-07 21:26:32 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-07 21:26:32 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-07 21:26:32 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 21:26:39 Total param: 477632
    2020-06-08 11:35:34 Epoch: 61, Train: 0.9746/0.0788 Test: 0.9110/0.3129
    
    
    
    ####################################################################
    3
    ####################################################################
    2020-06-07 21:27:39 #Conv=13 pretrained=False
    2020-06-07 21:27:39 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-07 21:27:39 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-07 21:27:39 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:1
    2020-06-07 21:27:39 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:False
    2020-06-07 21:27:46 Total param: 379328
    2020-06-08 17:37:21 Epoch: 88, Train: 1.0000/0.0035 Test: 0.8665/0.5050
    
    
    
    ####################################################################
    4
    ####################################################################
    2020-06-08 00:04:27 #Conv=13 pretrained=False
    2020-06-08 00:04:27 GCNNet1 #GNN1=1 in_dim=128 hidden_dims=[128] readout=mean
    2020-06-08 00:04:27 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-08 00:04:27 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-08 00:04:27 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-08 00:04:30 Total param: 362560
    2020-06-08 14:45:17 Epoch: 57, Train: 0.9665/0.1022 Test: 0.9040/0.3101
    
    2020-06-08 00:04:34 #Conv=13 pretrained=False
    2020-06-08 00:04:34 GCNNet1 #GNN1=3 in_dim=128 hidden_dims=[128, 128, 128] readout=mean
    2020-06-08 00:04:34 GCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-08 00:04:34 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-08 00:04:34 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-08 00:04:37 Total param: 396096
    2020-06-08 14:33:44 Epoch: 56, Train: 0.9652/0.1067 Test: 0.9064/0.3105
    
    2020-06-08 00:04:45 #Conv=13 pretrained=False
    2020-06-08 00:04:45 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-08 00:04:45 GCNNet2 #GNN2=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-08 00:04:45 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-08 00:04:45 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-08 00:04:48 Total param: 345792
    2020-06-08 13:39:24 Epoch: 53, Train: 0.9570/0.1329 Test: 0.9013/0.3103
    
    2020-06-08 00:04:49 #Conv=13 pretrained=False
    2020-06-08 00:04:49 GCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-08 00:04:49 GCNNet2 #GNN2=6 in_dim=128 hidden_dims=[128, 128, 128, 128, 128, 128] readout=mean
    2020-06-08 00:04:49 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:8 gpu:0
    2020-06-08 00:04:49 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-08 00:04:54 Total param: 412864
    2020-06-08 13:22:48 Epoch: 51, Train: 0.9543/0.1379 Test: 0.9045/0.3113
    
    
    
    ####################################################################
    5
    ####################################################################
    2020-06-07 23:55:51 #Conv=13 pretrained=False
    2020-06-07 23:55:51 GatedGCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-07 23:55:51 GatedGCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-07 23:55:51 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:6 gpu:1
    2020-06-07 23:55:51 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 23:55:54 Total param: 794176
    2020-06-08 16:02:30 Epoch: 61, Train: 0.9699/0.0906 Test: 0.9103/0.2937
    
    2020-06-07 23:55:56 #Conv=13 pretrained=False
    2020-06-07 23:55:56 GatedGCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=max
    2020-06-07 23:55:56 GatedGCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-07 23:55:56 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:6 gpu:1
    2020-06-07 23:55:56 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 23:55:59 Total param: 794176
    2020-06-08 15:07:03 Epoch: 57, Train: 0.9641/0.1101 Test: 0.9054/0.3132
    
    2020-06-07 23:56:02 #Conv=13 pretrained=False
    2020-06-07 23:56:02 GatedGCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-07 23:56:02 GatedGCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-07 23:56:02 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:6 gpu:1
    2020-06-07 23:56:02 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 23:56:05 Total param: 794176
    2020-06-08 21:16:15 Epoch: 80, Train: 0.9795/0.0614 Test: 0.9129/0.3053
    
    2020-06-07 23:56:05 #Conv=13 pretrained=False
    2020-06-07 23:56:05 GatedGCNNet1 #GNN1=2 in_dim=128 hidden_dims=[128, 128] readout=mean
    2020-06-07 23:56:05 GatedGCNNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-07 23:56:05 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:2 workers:6 gpu:1
    2020-06-07 23:56:05 down_ratio:2 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-07 23:56:10 Total param: 794176
    2020-06-08 14:33:09 Epoch: 55, Train: 0.9630/0.1130 Test: 0.9053/0.3121
    
    
    ####################################################################
    7
    ####################################################################
    2020-06-09 00:10:14 #Conv=6 pretrained=False
    2020-06-09 00:10:14 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=max
    2020-06-09 00:10:14 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-09 00:10:14 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:4 workers:8 gpu:0
    2020-06-09 00:10:14 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-09 00:10:17 Total param: 239296
    2020-06-09 21:53:15 Epoch: 79, Train: 0.9181/0.2406 Test: 0.8667/0.4023
    
    2020-06-09 00:10:28 #Conv=6 pretrained=False
    2020-06-09 00:10:28 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=mean
    2020-06-09 00:10:28 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-09 00:10:28 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:4 workers:8 gpu:0
    2020-06-09 00:10:28 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-09 00:10:33 Total param: 239296
    2020-06-09 22:18:26 Epoch: 80, Train: 0.9554/0.1317 Test: 0.8822/0.3974
    
    2020-06-09 00:10:24 #Conv=6 pretrained=False
    2020-06-09 00:10:24 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=mean
    2020-06-09 00:10:24 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=max
    2020-06-09 00:10:24 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:4 workers:8 gpu:0
    2020-06-09 00:10:24 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-09 00:10:28 Total param: 239296
    2020-06-09 19:16:58 Epoch: 69, Train: 0.9424/0.1687 Test: 0.8787/0.3664
    
    2020-06-09 00:10:18 #Conv=6 pretrained=False
    2020-06-09 00:10:18 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=max
    2020-06-09 00:10:18 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-09 00:10:18 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:4 workers:8 gpu:0
    2020-06-09 00:10:18 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-09 00:10:22 Total param: 239296
    2020-06-10 00:08:42 Epoch: 87, Train: 0.9492/0.1485 Test: 0.8734/0.4067
    
    
    
    ####################################################################
    9
    ####################################################################
    2020-06-10 00:32:30 #Conv=6 pretrained=False
    2020-06-10 00:32:30 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=mean
    2020-06-10 00:32:30 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-10 00:32:30 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:6 workers:8 gpu:0
    2020-06-10 00:32:30 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-10 00:32:34 Total param: 239296
    2020-06-10 13:45:58 Epoch: 78, Train: 0.9485/0.1509 Test: 0.8747/0.4032
    
    2020-06-10 00:32:27 #Conv=6 pretrained=False
    2020-06-10 00:32:27 GraphSageNet1 #GNN1=2 in_dim=64 hidden_dims=[128, 128] readout=mean
    2020-06-10 00:32:27 GraphSageNet2 #GNN2=4 in_dim=128 hidden_dims=[128, 128, 128, 128] readout=mean
    2020-06-10 00:32:27 ckpt:./ckpt2/dgl/4_DGL_CONV_CIFAR10/GCNNet3 is_sgd:True epochs:150 batch size:64 image size:32 sp size:5 workers:8 gpu:0
    2020-06-10 00:32:27 down_ratio:1 slic_max_iter:5 slic_sigma:1 slic_compactness:10 is_aug:True
    2020-06-10 00:32:34 Total param: 239296
    2020-06-10 10:34:12 Epoch: 63, Train: 0.9493/0.1493 Test: 0.8787/0.4047
    """

    _use_gpu = True

    # _model_conv, _model_gnn1, _model_gnn2 = None, None, None
    _data_root_path = '/private/alishuo/cifar10'
    # _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    # _data_root_path = "/home/ubuntu/ALISURE/data/cifar"
    _root_ckpt_dir = "./ckpt2/dgl/4_DGL_CONV_CIFAR10/{}".format("GCNNet3")
    _image_size = 32
    _train_print_freq = 200
    _test_print_freq = 100
    _num_workers = 8

    # 0 Batch size
    _batch_size = 64

    # 1 Optimizer
    _is_sgd, _epochs = True, 150
    # _lr = [[0, 0.1], [50, 0.01], [100, 0.001]]
    # _is_sgd, _epochs = False, 100
    # _lr = [[0, 0.01], [50, 0.001], [100, 0.0001]]

    # 2 Aug Data
    _is_aug = True

    # 3 SP
    _slic_compactness, _slic_sigma, _slic_max_iter = 10, 1, 5
    # _sp_size, _down_ratio = 2, 2
    # _sp_size, _down_ratio = 4, 1
    # _sp_size, _down_ratio = 2, 1
    # _sp_size, _down_ratio = 3, 1
    # _sp_size, _down_ratio = 5, 1
    _sp_size, _down_ratio = 6, 1

    # 4 GNN Number + Conv Number + Readout(mean, max, sum, topk)
    #
    _readout1, _readout2 = "mean", "mean"
    _lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]

    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64], no_conv_dim=64)  # C0
    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64], no_conv_dim=None)  # C1
    _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64], no_conv_dim=None)  # C2
    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, 64], no_conv_dim=None)  # C3
    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 128], no_conv_dim=None)  # C2P1C1
    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 128, 128], no_conv_dim=None)  # C2P1C2
    # _model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64, "M", 128, 128, 128], no_conv_dim=None)  # C2P1C3

    g1 = [256, 256]
    g2 = [512, 512, 512, 512]
    # _model_gnn1 = GCNNet1(64, g1, readout=_readout1)
    # _model_gnn2 = GCNNet2(g1[-1], g2, 10, readout=_readout2)
    # _model_gnn1 = GraphSageNet1(64, g1, readout=_readout1)
    # _model_gnn2 = GraphSageNet2(g1[-1], g2, 10, readout=_readout2)
    _model_gnn1 = GatedGCNNet1(64, g1, readout=_readout1)
    _model_gnn2 = GatedGCNNet2(g1[-1], g2, 10, readout=_readout2)

    _gpu_id = "0"

    Tools.print("ckpt:{} is_sgd:{} epochs:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _is_sgd, _epochs, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))
    Tools.print("down_ratio:{} slic_max_iter:{} slic_sigma:{} slic_compactness:{} is_aug:{}".format(
        _down_ratio, _slic_max_iter, _slic_sigma, _slic_compactness, _is_aug))

    runner = RunnerSPE(model_conv=_model_conv, model_gnn1=_model_gnn1,  model_gnn2=_model_gnn2,
                       use_gpu=_use_gpu, gpu_id=_gpu_id)
    pass
