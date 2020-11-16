import os
import cv2
import time
import glob
import torch
import skimage
import numpy as np
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg13_bn, vgg16_bn
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv


class CONVNet(nn.Module):

    def __init__(self, layer_num=14, out_dim=None, pretrained=True):  # 14, 23
        super().__init__()
        if out_dim:
            layers = [nn.Conv2d(3, out_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)
        else:
            self.features = vgg16_bn(pretrained=pretrained).features[0: layer_num]
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


class MySAGEConv1(SAGEConv):

    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, normalize=normalize, concat=concat, bias=bias, **kwargs)
        pass

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j + x_j

    pass


class MySAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=False, concat=True, bias=True, **kwargs):
        super().__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        in_channels = self.in_channels * 2 if self.concat else self.in_channels
        self.linear = nn.Linear(in_channels, self.out_channels, bias=bias)
        pass

    def forward(self, x, edge_index, edge_weight=None, size=None, res_n_id=None):
        if torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, None, 1, x.size(self.node_dim))
            pass
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j + x_j

    def update(self, aggr_out, x, res_n_id):
        if self.concat:
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        aggr_out = self.linear(aggr_out)
        aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

    pass


class MySAGEConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, normalize, concat, position=True, residual=True, gcn_num=1):
        super().__init__()
        self.position = position
        self.residual = residual
        self.gcn_num = gcn_num

        _in_dim = in_dim
        self.gcn = MySAGEConv(_in_dim, out_dim, normalize=normalize, concat=concat)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if self.gcn_num == 2:
            self.gcn2 = MySAGEConv(out_dim, out_dim, normalize=normalize, concat=concat)
            self.bn22 = nn.BatchNorm1d(out_dim)
            pass

        if self.position:
            self.pos = PositionEmbedding(2, _in_dim)
            if self.gcn_num == 2:
                self.pos2 = PositionEmbedding(2, out_dim)
                pass
            pass

        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x, data):
        identity = x
        out = x

        ##################################
        position_embedding = self.pos(data.edge_w) if self.position else None
        out = self.gcn(out, data.edge_index, edge_weight=position_embedding)
        out = self.bn2(out)

        if self.gcn_num == 2:
            out = self.relu(out)
            position_embedding = self.pos2(data.edge_w) if self.position else None
            out = self.gcn2(out, data.edge_index, edge_weight=position_embedding)
            out = self.bn22(out)
        ##################################

        if self.residual:
            if identity.size()[-1] == out.size()[-1]:
                out = out + identity
            pass

        out = self.relu(out)
        return out

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128], global_pool=global_mean_pool, normalize=False,
                 concat=False, residual=True, position=True, gcn_num=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.concat = concat
        self.residual = residual
        self.position = position

        embedding_dim = self.hidden_dims[0]
        self.embedding_h = nn.Linear(self.in_dim, embedding_dim, bias=False)

        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(MySAGEConvBlock(embedding_dim, hidden_dim, normalize=self.normalize,
                                                 concat=self.concat, residual=self.residual,
                                                 position=self.position, gcn_num=gcn_num))
            embedding_dim = hidden_dim
            pass

        self.global_pool = global_pool
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(hidden_nodes_feat, data)
            pass

        hg = self.global_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class SAGENet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128], normalize=False,
                 concat=False, residual=True, position=True, gcn_num=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.concat = concat
        self.residual = residual
        self.position = position

        embedding_dim = self.hidden_dims[0]
        self.embedding_h = nn.Linear(self.in_dim, embedding_dim, bias=False)

        self.gcn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(MySAGEConvBlock(embedding_dim, hidden_dim, normalize=self.normalize,
                                                 concat=self.concat, residual=self.residual,
                                                 position=self.position, gcn_num=gcn_num))
            embedding_dim = hidden_dim
            pass
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(hidden_nodes_feat, data)
            pass

        return hidden_nodes_feat

    pass


class PositionEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.position_embedding_1 = nn.Linear(in_dim, out_dim // 2, bias=False)
        self.position_embedding_2 = nn.Linear(out_dim // 2, out_dim, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, x):
        out = self.position_embedding_1(x)
        out = self.relu(out)
        out = self.position_embedding_2(out)
        return out

    pass


class AttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10, global_pool=global_max_pool, is_attention=True):
        super().__init__()
        self.global_pool = global_pool

        self.is_attention = is_attention
        if self.is_attention:
            self.attention = nn.Linear(in_dim, 1, bias=False)
            pass

        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data):
        x = data.x

        if self.is_attention:
            x_att = torch.sigmoid(self.attention(x))
            x = (x_att * x + x) / 2

        hg = self.global_pool(x, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=14, normalize=True, residual=True, concat=True, pretrained=True,
                 hidden_dims1=[128, 128], hidden_dims2=[128, 128, 128, 128],
                 global_pool_1=global_mean_pool, global_pool_2=global_max_pool, gcn_num=1):
        super().__init__()

        self.model_conv = CONVNet(layer_num=conv_layer_num, pretrained=pretrained)  # 6, 13
        self.attention_class = AttentionClass(in_dim=hidden_dims2[-1], n_classes=1000,
                                              global_pool=global_pool_2, is_attention=True)

        assert conv_layer_num == 14 or conv_layer_num == 23
        in_dim_which = -3 if conv_layer_num == 14 else -2

        self.model_gnn1 = SAGENet1(in_dim=self.model_conv.features[in_dim_which].num_features,
                                   hidden_dims=hidden_dims1, normalize=normalize, residual=residual,
                                   position=Param.position, concat=concat, global_pool=global_pool_1, gcn_num=gcn_num)
        self.model_gnn2 = SAGENet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=hidden_dims2,
                                   position=Param.position, normalize=normalize,
                                   residual=residual, concat=concat, gcn_num=gcn_num)
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
        gcn2_feature = self.model_gnn2.forward(batched_graph)

        batched_graph.x = gcn2_feature
        logits = self.attention_class.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self):
        self.model = MyGCNNet(conv_layer_num=Param.conv_layer_num, normalize=Param.normalize,
                              residual=Param.residual, concat=Param.concat, hidden_dims1=Param.hidden_dims1,
                              hidden_dims2=Param.hidden_dims2, pretrained=False, gcn_num=Param.gcn_num,
                              global_pool_1=Param.global_pool_1, global_pool_2=Param.global_pool_2)
        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        pass

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


"""
2020-10-21 12:18:43 Epoch:76, Train:0.7431-0.9209/1.0510 Test:0.6974-0.8935/1.2385
"""


class Param(object):
    position = True
    # position = False

    has_linear_in_block1 = False
    has_linear_in_block2 = False

    gcn_num = 2

    sp_size, down_ratio, conv_layer_num = 4, 4, 23  # GCNNet-C2PC2PC3

    hidden_dims1 = [256, 256]
    hidden_dims2 = [512, 512, 512]

    global_pool_1, global_pool_2 = global_mean_pool, global_mean_pool

    concat, residual, normalize = True, True, True
    pass


if __name__ == '__main__':
    runner = RunnerSPE()
    pass
