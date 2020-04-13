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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4, sp_ve_size=6):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path
        # self.transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                                      transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #                                      transforms.RandomGrayscale(p=0.2),
        #                                      transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.transform = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)

        # 3. Super Pixel
        self.image_size = image_size
        self.sp_size = sp_size
        self.sp_ve_size = sp_ve_size
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        img = np.asarray(img)
        graph, target = self.get_sp_info(img, target)
        return graph, img, target

    def get_sp_info(self, img, target, is_add_self=False):
        # 3. Super Pixel
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size, super_pixel_size=self.sp_size)
        _, super_pixel_info, adjacency_info = deal_super_pixel.run()

        # Resize Super Pixel
        _now_data_list = [cv2.resize(super_pixel_info[key]["data2"] / 255, (self.sp_ve_size, self.sp_ve_size),
                                     interpolation=cv2.INTER_NEAREST) for key in super_pixel_info]
        _now_shape_list = [np.expand_dims(cv2.resize(
            super_pixel_info[key]["label"] / 1, (self.sp_ve_size, self.sp_ve_size),
            interpolation=cv2.INTER_NEAREST), axis=-1) for key in super_pixel_info]
        net_data = np.transpose(_now_data_list, axes=(0, 3, 1, 2))
        net_shape = np.transpose(_now_shape_list, axes=(0, 3, 1, 2))

        # Node
        pos, area, size = [], [], []
        for sp_i in range(len(super_pixel_info)):
            _size = super_pixel_info[sp_i]["size"]
            _area = super_pixel_info[sp_i]["area"]

            size.append([_size])
            area.append(_area)
            pos.append([_area[1] - _area[0], _area[3] - _area[2]])
            pass
        pos = np.asarray(pos) / self.image_size
        size = np.asarray(size)
        area = np.asarray(area)

        # Graph
        graph = dgl.DGLGraph()

        # Node Add
        graph.add_nodes(net_data.shape[0])
        graph.ndata['data'] = torch.from_numpy(net_data).float()
        graph.ndata['shape'] = torch.from_numpy(net_shape).float()
        graph.ndata['pos'] = torch.from_numpy(pos).float()
        graph.ndata['size'] = torch.from_numpy(size).float()
        graph.ndata['area'] = torch.from_numpy(area).float()

        # Edge
        edge_index, edge_w = [], []
        for edge_i in range(len(adjacency_info)):
            edge_index.append([adjacency_info[edge_i][0], adjacency_info[edge_i][1]])
            edge_w.append(adjacency_info[edge_i][2])
            pass
        edge_index = np.asarray(edge_index)
        edge_w = np.asarray(edge_w)

        # Edge Add
        if not is_add_self:
            graph.add_edges(edge_index[:, 0], edge_index[:, 1])
            graph.edata['feat'] = torch.from_numpy(edge_w).unsqueeze(1).float()
        else:
            non_self_edges_idx = edge_index[:, 0] != edge_index[:, 1]
            graph.add_edges(edge_index[:, 0][non_self_edges_idx], edge_index[:, 1][non_self_edges_idx])
            edge_w_new = list(edge_w[non_self_edges_idx])

            _nodes = np.arange(graph.number_of_nodes())
            graph.add_edges(_nodes, _nodes)
            edge_w_new.extend([1.0] * graph.number_of_nodes())
            edge_w = np.asarray(edge_w_new)
            graph.edata['feat'] = torch.from_numpy(edge_w).unsqueeze(1).float()
            pass

        return graph, target

    @staticmethod
    def collate_fn(samples):
        graphs, imgs, labels = map(list, zip(*samples))
        imgs = torch.tensor(np.array(imgs))
        labels = torch.tensor(np.array(labels))

        nodes_num = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        edges_num = [graphs[i].number_of_edges() for i in range(len(graphs))]
        nodes_num_norm = [torch.zeros((num, 1)).fill_(1. / float(num)) for num in nodes_num]
        edges_num_norm = [torch.zeros((num, 1)).fill_(1. / float(num)) for num in edges_num]
        nodes_num_norm_sqrt = torch.cat(nodes_num_norm).sqrt()
        edges_num_norm_sqrt = torch.cat(edges_num_norm).sqrt()

        batched_graph = dgl.batch(graphs)
        return batched_graph, imgs, labels, nodes_num_norm_sqrt, edges_num_norm_sqrt

    pass


class MLPNet(nn.Module):

    def __init__(self, layer=4, in_dim=32, residual=True):
        super().__init__()

        self.L = layer
        self.in_dim = in_dim
        self.hidden_dim = 168
        self.n_classes = 10
        self.dropout = 0.0
        self.in_feat_dropout = 0.0
        self.gated = False

        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        feat_mlp_modules = [nn.Linear(self.in_dim, self.hidden_dim, bias=True), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(self.L - 1):  # L=4
            feat_mlp_modules.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(self.dropout))  # 168, 168
            pass
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)  # 168, 168
            pass

        self.readout_mlp = MLPReadout(self.hidden_dim, self.n_classes)  # MLP: 3
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.in_feat_dropout(nodes_feat)
        h = self.feat_mlp(h)

        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h
            graphs.ndata['h'] = h
            hg = dgl.sum_nodes(graphs, 'h')
        else:
            graphs.ndata['h'] = h
            hg = dgl.mean_nodes(graphs, 'h')
            pass

        logits = self.readout_mlp(hg)
        return logits

    pass


class GCNNet(nn.Module):

    def __init__(self, layer=4, in_dim=32, residual=True):
        super().__init__()
        self.L = layer
        self.readout = "mean"
        self.in_dim = in_dim
        self.hidden_dim = 146
        self.out_dim = 146
        self.n_classes = 10
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        self.residual = residual

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        self.layers = nn.ModuleList([GCNLayer(self.hidden_dim, self.hidden_dim, F.relu, self.dropout, self.graph_norm,
                                              self.batch_norm, self.residual) for _ in range(self.L - 1)])

        self.layers.append(GCNLayer(self.hidden_dim, self.out_dim, F.relu, self.dropout,
                                    self.graph_norm, self.batch_norm, self.residual))

        self.readout_mlp = MLPReadout(self.out_dim, self.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        hidden_nodes_feat = self.in_feat_dropout(hidden_nodes_feat)
        for conv in self.layers:
            hidden_nodes_feat = conv(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = self.readout_fn(self.readout, graphs, 'h')

        logits = self.readout_mlp(hg)
        return logits

    @staticmethod
    def readout_fn(readout, graphs, h):
        if readout == "sum":
            hg = dgl.sum_nodes(graphs, h)
        elif readout == "max":
            hg = dgl.max_nodes(graphs, h)
        elif readout == "mean":
            hg = dgl.mean_nodes(graphs, h)
        else:
            hg = dgl.mean_nodes(graphs, h)  # default readout is mean nodes
        return hg

    pass


class GraphSageNet(nn.Module):

    def __init__(self, layer=4, in_dim=32, residual=True):
        super().__init__()
        self.L = layer
        self.out_dim = 108
        self.residual = residual
        self.in_dim = in_dim
        self.hidden_dim = 108
        self.n_classes = 10
        self.in_feat_dropout = 0.0
        self.sage_aggregator = "meanpool"
        self.readout = "mean"
        self.dropout = 0.0

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(self.hidden_dim, self.hidden_dim, F.relu, self.dropout,
                                                    self.sage_aggregator, self.residual) for _ in range(self.L - 1)])
        self.layers.append(GraphSageLayer(self.hidden_dim, self.out_dim, F.relu,
                                          self.dropout, self.sage_aggregator, self.residual))
        self.readout_mlp = MLPReadout(self.out_dim, self.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(graphs, h, nodes_num_norm_sqrt)
        graphs.ndata['h'] = h
        hg = self.readout_fn(self.readout, graphs, 'h')

        logits = self.readout_mlp(hg)
        return logits

    @staticmethod
    def readout_fn(readout, graphs, h):
        if readout == "sum":
            hg = dgl.sum_nodes(graphs, h)
        elif readout == "max":
            hg = dgl.max_nodes(graphs, h)
        elif readout == "mean":
            hg = dgl.mean_nodes(graphs, h)
        else:
            hg = dgl.mean_nodes(graphs, h)  # default readout is mean nodes
        return hg

    pass


class GATNet(nn.Module):

    def __init__(self, layer=4, in_dim=32, residual=True):
        super().__init__()

        self.L = layer
        self.out_dim = 152
        self.residual = residual
        self.readout = "mean"
        self.in_dim = in_dim
        self.hidden_dim = 19
        self.n_heads = 8
        self.n_classes = 10
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim * self.n_heads)
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        self.layers = nn.ModuleList([GATLayer(self.hidden_dim * self.n_heads, self.hidden_dim, self.n_heads,
                                              self.dropout, self.graph_norm, self.batch_norm,
                                              self.residual) for _ in range(self.L - 1)])
        self.layers.append(GATLayer(self.hidden_dim * self.n_heads, self.out_dim, 1, self.dropout,
                                    self.graph_norm, self.batch_norm, self.residual))

        self.readout_mlp = MLPReadout(self.out_dim, self.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(graphs, h, nodes_num_norm_sqrt)
        graphs.ndata['h'] = h
        hg = self.readout_fn(self.readout, graphs, 'h')

        logits = self.readout_mlp(hg)
        return logits

    @staticmethod
    def readout_fn(readout, graphs, h):
        if readout == "sum":
            hg = dgl.sum_nodes(graphs, h)
        elif readout == "max":
            hg = dgl.max_nodes(graphs, h)
        elif readout == "mean":
            hg = dgl.mean_nodes(graphs, h)
        else:
            hg = dgl.mean_nodes(graphs, h)  # default readout is mean nodes
        return hg

    pass


class GatedGCNNet(nn.Module):

    def __init__(self, layer=4, in_dim=32, residual=True):
        super().__init__()

        self.in_feat_dropout = 0.0
        self.dropout = 0.0

        self.L = layer
        self.in_dim = in_dim
        self.in_dim_edge = 1
        self.n_classes = 10
        self.out_dim = 70
        self.residual = residual
        self.readout = "mean"
        self.hidden_dim = 70
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)
        self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dim)
        self.layers = nn.ModuleList([GatedGCNLayer(self.hidden_dim, self.hidden_dim, self.dropout, self.graph_norm,
                                                   self.batch_norm, self.residual) for _ in range(self.L - 1)])
        self.layers.append(GatedGCNLayer(self.hidden_dim, self.out_dim, self.dropout,
                                         self.graph_norm, self.batch_norm, self.residual))
        self.readout_mlp = MLPReadout(self.out_dim, self.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        # input embedding
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        # convnets
        for conv in self.layers:
            h, e = conv(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = self.readout_fn(self.readout, graphs, 'h')

        logits = self.readout_mlp(hg)
        return logits

    @staticmethod
    def readout_fn(readout, graphs, h):
        if readout == "sum":
            hg = dgl.sum_nodes(graphs, h)
        elif readout == "max":
            hg = dgl.max_nodes(graphs, h)
        elif readout == "mean":
            hg = dgl.mean_nodes(graphs, h)
        else:
            hg = dgl.mean_nodes(graphs, h)  # default readout is mean nodes
        return hg

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


class EmbeddingNetCIFARSmallNorm(nn.Module):

    def __init__(self, has_sigmoid=True, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_sigmoid = has_sigmoid

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
        self.texture_up1 = nn.UpsamplingBilinear2d(scale_factor=3)  # 3
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
        shape_out = self.sigmoid(shape_out) if self.has_sigmoid else shape_out

        texture_d0 = torch.cat([texture_norm, shape_norm], dim=1)
        texture_d1 = self.texture_up1(self.texture_conv1(texture_d0))
        texture_d2 = self.texture_up2(self.texture_conv22(self.texture_conv21(texture_d1)))
        texture_out = self.texture_out(self.texture_conv32(self.texture_conv31(texture_d2)))
        texture_out = self.sigmoid(texture_out) if self.has_sigmoid else texture_out

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


class MyGCNNet(nn.Module):

    def __init__(self, gcn_model, layer=4, in_dim=32, residual=True, has_sigmoid=True):
        super().__init__()
        self.in_dim = in_dim
        self.ve_model = EmbeddingNetCIFARSmallNorm(has_sigmoid=has_sigmoid)
        self.gcn_model = gcn_model(layer=layer, in_dim=self.in_dim, residual=residual)
        pass

    def forward(self, graphs, nodes_data, nodes_pos, nodes_size, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        shape_feature, texture_feature, shape_out, texture_out = self.ve_model.forward(nodes_data)

        if self.in_dim == 32:
            nodes_feat = torch.cat([shape_feature, texture_feature], dim=1)
        else:
            nodes_feat = torch.cat([shape_feature, texture_feature, nodes_pos, nodes_size], dim=1)

        graphs.ndata['feat'] = nodes_feat

        logits = self.gcn_model.forward(graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        return shape_out, texture_out, logits

    pass


class RunnerSPE(object):

    def __init__(self, gcn_model=GCNNet, data_root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=64,
                 layer=4, in_dim=32, residual=True, has_sigmoid=True, sp_size=4, sp_ve_size=6, image_size=32,
                 is_mse_loss=True, root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)
        self.is_mse_loss = is_mse_loss

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True,
                                       image_size=image_size, sp_size=sp_size, sp_ve_size=sp_ve_size)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False,
                                      image_size=image_size, sp_size=sp_size, sp_ve_size=sp_ve_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(gcn_model, layer=layer, in_dim=in_dim,
                              residual=residual, has_sigmoid=has_sigmoid).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.0)

        self.loss_class = nn.CrossEntropyLoss().to(self.device)
        if self.is_mse_loss:
            self.loss_shape = nn.MSELoss().to(self.device)
            self.loss_texture = nn.MSELoss().to(self.device)
        else:
            self.loss_shape = nn.BCELoss().to(self.device)
            self.loss_texture = nn.BCELoss().to(self.device)
            pass

        Tools.print("Total param: {} gcn_model: {}".format(self._view_model_param(self.model), gcn_model))
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
        epoch_loss, epoch_loss_shape, epoch_loss_texture, epoch_loss_class, epoch_train_acc, nb_data = 0, 0, 0, 0, 0, 0
        for i, (batch_graphs, batch_imgs, batch_labels,
                batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(self.train_loader):
            batch_nodes_data = batch_graphs.ndata['data'].to(self.device)  # num x feat
            batch_nodes_shape = batch_graphs.ndata['shape'].to(self.device)  # num x feat
            batch_nodes_pos = batch_graphs.ndata['pos'].to(self.device)
            batch_nodes_size = batch_graphs.ndata['size'].to(self.device)
            batch_edges_feat = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.long().to(self.device)
            batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.device)  # num x 1
            batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.device)

            self.optimizer.zero_grad()
            shape_out, texture_out, logits = self.model.forward(
                batch_graphs, batch_nodes_data, batch_nodes_pos, batch_nodes_size,
                batch_edges_feat, batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
            loss, loss_shape, loss_texture, loss_class = self._loss_total(
                shape_out, texture_out, logits, batch_nodes_shape, batch_nodes_data, batch_labels)
            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_loss_shape += loss_shape.detach().item()
            epoch_loss_texture += loss_texture.detach().item()
            epoch_loss_class += loss_class.detach().item()
            epoch_train_acc += self._accuracy(logits, batch_labels)

            if i % print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} shape={:4f}/{:4f} "
                            "texture={:4f}/{:4f} class={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.train_loader), epoch_loss / (i + 1), loss.detach().item(), epoch_loss_shape / (i + 1),
                    loss_shape.detach().item(), epoch_loss_texture / (i + 1), loss_texture.detach().item(),
                                               epoch_loss_class / (i + 1), loss_class.detach().item(),
                                               epoch_train_acc / nb_data))
                pass
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self, print_freq=50):
        self.model.eval()

        Tools.print()
        epoch_test_acc, nb_data = 0, 0
        epoch_test_loss, epoch_test_loss_shape, epoch_test_loss_texture, epoch_test_loss_class = 0, 0, 0, 0
        with torch.no_grad():
            for i, (batch_graphs, batch_imgs, batch_labels,
                    batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(self.test_loader):
                batch_nodes_data = batch_graphs.ndata['data'].to(self.device)  # num x feat
                batch_nodes_shape = batch_graphs.ndata['shape'].to(self.device)  # num x feat
                batch_nodes_pos = batch_graphs.ndata['pos'].to(self.device)
                batch_nodes_size = batch_graphs.ndata['size'].to(self.device)
                batch_edges_feat = batch_graphs.edata['feat'].to(self.device)
                batch_labels = batch_labels.long().to(self.device)
                batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.device)
                batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.device)

                shape_o, texture_o, logits = self.model.forward(
                    batch_graphs, batch_nodes_data, batch_nodes_pos, batch_nodes_size,
                    batch_edges_feat, batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
                loss, loss_shape, loss_texture, loss_class = self._loss_total(
                    shape_o, texture_o, logits, batch_nodes_shape, batch_nodes_data, batch_labels)

                nb_data += batch_labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_loss_shape += loss_shape.detach().item()
                epoch_test_loss_texture += loss_texture.detach().item()
                epoch_test_loss_class += loss_class.detach().item()
                epoch_test_acc += self._accuracy(logits, batch_labels)

                if i % print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} shape={:4f}/{:4f} "
                                "texture={:4f}/{:4f} class={:4f}/{:4f} acc={:4f}".format(
                        i, len(self.test_loader), epoch_test_loss / (i + 1), loss.detach().item(),
                                                  epoch_test_loss_shape / (i + 1), loss_shape.detach().item(),
                                                  epoch_test_loss_texture / (i + 1), loss_texture.detach().item(),
                                                  epoch_test_loss_class / (i + 1), loss_class.detach().item(),
                                                  epoch_test_acc / nb_data))
                    pass
                pass
            pass

        epoch_test_loss /= (len(self.test_loader) + 1)
        epoch_test_acc /= nb_data
        return epoch_test_loss, epoch_test_acc

    def _loss_total(self, shape_out, texture_out, logits, batch_nodes_shape, batch_nodes_data, batch_labels):
        loss_shape = self.loss_shape(shape_out, batch_nodes_shape)

        _positions = batch_nodes_data >= 0
        loss_texture = self.loss_texture(texture_out[_positions], batch_nodes_data[_positions])
        loss_class = self.loss_class(logits, batch_labels)
        loss_total = loss_shape + loss_texture + loss_class
        return loss_total, loss_shape, loss_texture, loss_class

    def _lr(self, epoch):
        if epoch == 25:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
            pass

        if epoch == 50:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0005
            pass

        if epoch == 75:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
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
    # layer, size, size, image size, batch size
    # 强数据增强+LR。不确定以下两个哪个带Sigmoid
    GCN  No Sigmoid 4 4 6 32 64 2020-04-07 02:50:57 Epoch: 75, lr=0.0000, Train: 0.5148/1.4100 Test: 0.5559/1.3145
    GCN Has Sigmoid 4 4 6 32 64 2020-04-07 07:35:40 Epoch: 72, lr=0.0000, Train: 0.5354/1.3428 Test: 0.5759/1.2394
    GCN  No Sigmoid 4 4 6 32 64 2020-04-08 06:36:51 Epoch: 70, lr=0.0000, Train: 0.5099/1.4281 Test: 0.5505/1.3224
    GCN Has Sigmoid 4 4 6 32 64 2020-04-08 07:24:54 Epoch: 73, lr=0.0001, Train: 0.5471/1.3164 Test: 0.5874/1.2138

    # 原始:数据增强+LR
    GCN           No Sigmoid 4 4 6 32 64 2020-04-08 06:24:55 Epoch: 98, lr=0.0001, Train: 0.6696/0.9954 Test: 0.6563/1.0695
    GCN          Has Sigmoid 4 4 6 32 64 2020-04-08 15:41:33 Epoch: 97, lr=0.0001, Train: 0.7781/0.6535 Test: 0.7399/0.8137
    GraphSageNet Has Sigmoid 4 4 6 32 64 2020-04-08 23:31:25 Epoch: 88, lr=0.0001, Train: 0.8074/0.5703 Test: 0.7612/0.7322
    GatedGCNNet  Has Sigmoid 4 4 6 32 64 2020-04-10 03:55:12 Epoch: 92, lr=0.0001, Train: 0.8401/0.4779 Test: 0.7889/0.6741
    
    # 原始:数据增强+LR
    GCN          Has Sigmoid 3 4 6 32 64 2020-04-11 21:20:56 Epoch: 94, lr=0.0001, Train: 0.7638/0.6942 Test: 0.7256/0.8472
    GCN          Has Sigmoid 5 4 6 32 64 2020-04-11 21:59:27 Epoch: 99, lr=0.0001, Train: 0.7523/0.7447 Test: 0.7180/0.8634
    GCN          Has Sigmoid 4 3 6 32 64 2020-04-12 21:31:36 Epoch: 78, lr=0.0001, Train: 0.7042/0.8552 Test: 0.6743/0.9571
    GCN          Has Sigmoid 3 6 6 32 64 2020-04-12 09:37:31 Epoch: 84, lr=0.0001, Train: 0.6293/1.0990 Test: 0.6210/1.1393
    GCN          Has Sigmoid 4 6 6 32 64 2020-04-12 04:10:02 Epoch: 93, lr=0.0001, Train: 0.6244/1.1293 Test: 0.6168/1.1562
    """
    # _gcn_model = GCNNet
    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("GCNNet")
    # _num_workers = 2
    # _use_gpu = False
    # _gpu_id = "1"

    # _gcn_model = MLPNet
    _gcn_model = GCNNet
    # _gcn_model = GATNet
    # _gcn_model = GraphSageNet
    # _gcn_model = GatedGCNNet
    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    _root_ckpt_dir = "./ckpt2/dgl/3_DGL_E2E_Layer_Size/{}-wa-lr-sigmoid".format("GCNNet")
    _has_sigmoid = True
    _is_mse_loss = True
    _batch_size = 64
    # _in_dim = 32
    _in_dim = 32 + 3
    _sp_size = 4
    _sp_ve_size = 6
    _image_size = 32
    _layer = 4
    _num_workers = 8
    _use_gpu = True
    _gpu_id = "1"

    Tools.print("ckpt:{}, sigmoid:{}, mse:{}, layer:{}, workers:{}, gpu:{}, model:{}, "
                "sp size:{}, sp ve size:{}, image size:{}, batch size:{}, in dim:{}".format(
        _root_ckpt_dir, _has_sigmoid, _is_mse_loss, _layer, _num_workers,
        _gpu_id, _gcn_model, _sp_size, _sp_ve_size, _image_size, _batch_size, _in_dim))

    runner = RunnerSPE(gcn_model=_gcn_model, data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       is_mse_loss=_is_mse_loss, has_sigmoid=_has_sigmoid, layer=_layer, batch_size=_batch_size,
                       sp_size=_sp_size, sp_ve_size=_sp_ve_size, image_size=_image_size, in_dim=_in_dim,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(100)
    pass
