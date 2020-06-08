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


class DealSuperPixel(object):

    def __init__(self, image_data, ds_image_size=224, super_pixel_size=14,
                 slic_compactness=10, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num, start_label=0,
                                         compactness=slic_compactness, sigma=slic_sigma, max_iter=slic_max_iter)
        # self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
        #                                  sigma=slic_sigma, max_iter=slic_max_iter)

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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32,
                 sp_size=4, slic_compactness=10, slic_sigma=1, slic_max_iter=5, down_ratio=1, is_aug=True):
        super().__init__()
        self.sp_size = sp_size
        self.slic_compactness = slic_compactness
        self.slic_sigma = slic_sigma
        self.slic_max_iter = slic_max_iter

        self.is_aug = is_aug

        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // down_ratio
        self.data_root_path = data_root_path

        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.transform = self.transform if self.is_aug else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        img_data = transforms.Compose([transforms.ToTensor(), normalize])(np.asarray(img)).unsqueeze(dim=0)

        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data)
        return graph, pixel_graph, img_data, target

    def get_sp_info(self, img):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size_for_sp,
                                          super_pixel_size=self.sp_size, slic_compactness=self.slic_compactness,
                                          slic_sigma=self.slic_sigma, slic_max_iter=self.slic_max_iter)
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


class CONVNet(nn.Module):

    def __init__(self, layer_num=6, out_dim=None, pretrained=True):  # 6, 13
        super().__init__()
        self.pretrained = pretrained
        if out_dim:
            layers = [nn.Conv2d(3, out_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)
        else:
            self.features = vgg13_bn(pretrained=self.pretrained).features[0: layer_num]

        Tools.print("#Conv={} pretrained={}".format(len(self.features), self.pretrained))
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

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', down_ratio=1, is_aug=True,
                 model_conv=None, model_gnn1=None, model_gnn2=None, lr=[[0, 0.1], [50, 0.01], [100, 0.001]],
                 slic_compactness=10, slic_sigma=1, slic_max_iter=5,
                 batch_size=64, image_size=32, sp_size=4, train_print_freq=100, test_print_freq=50,
                 is_sgd=True, root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                       is_train=True, image_size=image_size,
                                       sp_size=sp_size, slic_compactness=slic_compactness,
                                       slic_sigma=slic_sigma, slic_max_iter=slic_max_iter, is_aug=is_aug)
        self.test_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                      is_train=False, image_size=image_size,
                                      sp_size=sp_size, slic_compactness=slic_compactness,
                                      slic_sigma=slic_sigma, slic_max_iter=slic_max_iter, is_aug=is_aug)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        if model_gnn1 and model_gnn2:
            self.model = MyGCNNet(model_conv=model_conv, model_gnn1=model_gnn1, model_gnn2=model_gnn2).to(self.device)
        else:
            self.model = MyGCNNet().to(self.device)

        self.lr_s = lr
        if is_sgd:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr_s[0][1], momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=0.0)

        self.loss_class = nn.CrossEntropyLoss().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
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

            epoch_loss, epoch_train_acc = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc = self.test()

            Tools.print('Epoch: {:02d}, Train: {:.4f}/{:.4f} Test: {:.4f}/{:.4f}'.format(
                epoch, epoch_train_acc, epoch_loss, epoch_test_acc, epoch_test_loss))
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
    """

    _use_gpu = True

    # _model_conv, _model_gnn1, _model_gnn2 = None, None, None
    _data_root_path = '/private/alishuo/cifar10'
    # _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
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
    _sp_size, _down_ratio = 4, 1

    # 4 GNN Number + Conv Number + Readout(mean, max, sum, topk)
    #
    # _readout1, _readout2 = "max", "max"
    # _lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]
    # _readout1, _readout2 = "max", "mean"
    # _lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]
    # _readout1, _readout2 = "mean", "max"
    # _lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]
    _readout1, _readout2 = "mean", "mean"
    _lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]
    #
    _model_conv = CONVNet(layer_num=6, pretrained=False)
    # _model_conv = CONVNet(layer_num=13, pretrained=False)
    # _model_gnn1 = GCNNet1(64, [128, 128], readout=_readout1)
    # _model_gnn2 = GCNNet2(128, [128, 128, 128, 128], 10, readout=_readout2)
    _model_gnn1 = GraphSageNet1(64, [128, 128], readout=_readout1)
    _model_gnn2 = GraphSageNet2(128, [128, 128, 128, 128], 10, readout=_readout2)

    _gpu_id = "0"
    # _gpu_id = "1"

    Tools.print("ckpt:{} is_sgd:{} epochs:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _is_sgd, _epochs, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))
    Tools.print("down_ratio:{} slic_max_iter:{} slic_sigma:{} slic_compactness:{} is_aug:{}".format(
        _down_ratio, _slic_max_iter, _slic_sigma, _slic_compactness, _is_aug))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, down_ratio=_down_ratio,
                       model_conv=_model_conv, model_gnn1=_model_gnn1,  model_gnn2=_model_gnn2, lr=_lr,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id, is_aug=_is_aug,
                       slic_compactness=_slic_compactness, slic_sigma=_slic_sigma, slic_max_iter=_slic_max_iter)
    runner.train(_epochs, start_epoch=0)
    pass
