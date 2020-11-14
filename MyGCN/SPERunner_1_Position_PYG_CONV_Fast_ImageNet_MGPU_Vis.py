import os
import cv2
import time
import glob
import torch
import skimage
import numpy as np
from PIL import Image
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
        edge_index, edge_position, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(self.segment == i, selem=skimage.morphology.square(3))
            _position = []
            for sp_id in np.unique(self.segment[_now_adj]):
                if sp_id != i:
                    edge_index.append([i, sp_id])
                    _position.append([self.region_props[i][0][0] - self.region_props[sp_id][0][0],
                                      self.region_props[i][0][1] - self.region_props[sp_id][0][1]])
                pass
            edge_position.extend(_position)

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            _dis = _dis <= 2
            pixel_edge_index = np.argwhere(_dis)
            pixel_edge_position = [[pixel_data_where[edge_i][1] - pixel_data_where[edge_j][1],
                                    pixel_data_where[edge_i][2] - pixel_data_where[edge_j][2]]
                                   for edge_i, edge_j in pixel_edge_index]
            pixel_edge_w = np.asarray(pixel_edge_position)

            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w])
            pass

        sp_adj = [np.asarray(edge_index), np.asarray(edge_position)]
        return self.segment, sp_adj, pixel_adj

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\\data\\ImageNet\\ILSVRC2015\\Data\\CLS-LOC', down_ratio=4,
                 is_train=True, image_size=224, sp_size=11, train_split="train", test_split="val"):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // down_ratio
        self.data_root_path = data_root_path

        self.transform_train = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomCrop(self.image_size),
                                                   transforms.RandomHorizontalFlip()])
        # self.transform_train = transforms.Compose([transforms.RandomResizedCrop(self.image_size),
        #                                            transforms.RandomHorizontalFlip()])
        self.transform_test = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(self.image_size)])

        _test_dir = os.path.join(self.data_root_path, test_split)
        _train_dir = os.path.join(self.data_root_path, train_split)
        _transform = self.transform_train if self.is_train else self.transform_test

        self.data_set = datasets.ImageFolder(root=_train_dir if self.is_train else _test_dir, transform=_transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_path = self.data_set.samples[idx][0]
        img, target = self.data_set.__getitem__(idx)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), normalize])(np.asarray(img)).unsqueeze(dim=0)

        img_vis_data = np.asarray(img)
        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data, target)
        return graph, pixel_graph, img_data, target, img_path, img_vis_data

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
                     edge_w=torch.from_numpy(sp_adj[1]).float())
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([target]),
                               edge_w=torch.from_numpy(super_pixel[2]).float())
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph

    @staticmethod
    def collate_fn(samples):
        gpu_num = torch.cuda.device_count()
        one_num = len(samples) // gpu_num
        samples_list = [samples[i * one_num:(i + 1) * one_num] for i in range(gpu_num)] if one_num > 0 else [samples]

        graphs_list, pixel_graphs_list, images_list, labels_list = [], [], [], []
        images_path_list, image_small_data_list = [], []
        for samples_now in samples_list:
            graphs, pixel_graphs, images, labels, images_path, image_small_data = map(list, zip(*samples_now))

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

            images_path_list.append(images_path)
            image_small_data_list.append(image_small_data)
            images_list.append(images)
            labels_list.append(labels)
            graphs_list.append(batched_graph)
            pixel_graphs_list.append(batched_pixel_graph)
            pass

        return images_list, labels_list, graphs_list, pixel_graphs_list, images_path_list, image_small_data_list

    pass


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


class MySAGEConv(SAGEConv):

    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, normalize=normalize, concat=concat, bias=bias, **kwargs)
        pass

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight + x_j
        # return x_j if edge_weight is None else edge_weight * x_j
        return x_j if edge_weight is None else edge_weight * x_j + x_j

    pass


class MySAGEConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, normalize, concat,
                 position=True, residual=True, gcn_num=1, has_linear_in_block=False):
        super().__init__()
        self.position = position
        self.residual = residual
        self.gcn_num = gcn_num
        self.has_linear_in_block = has_linear_in_block

        _in_dim = in_dim
        if self.has_linear_in_block:
            self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
            self.bn1 = nn.BatchNorm1d(out_dim)
            _in_dim = out_dim
            pass

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

        if self.has_linear_in_block:
            self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
            self.bn3 = nn.BatchNorm1d(out_dim)

        if self.has_linear_in_block:
            if in_dim != out_dim:
                self.linear0 = nn.Linear(in_dim, out_dim, bias=False)
                pass
            pass

        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x, data):
        identity = x

        if self.has_linear_in_block:
            out = self.relu(self.bn1(self.linear1(x)))
        else:
            out = x

        ##################################
        position_embedding = self.pos(data.edge_w) if self.position else None
        out = self.gcn(out, data.edge_index, edge_weight=position_embedding)
        out = self.relu(self.bn2(out))

        if self.gcn_num == 2:
            position_embedding = self.pos2(data.edge_w) if self.position else None
            out = self.gcn2(out, data.edge_index, edge_weight=position_embedding)
            out = self.relu(self.bn22(out))
        ##################################

        if self.has_linear_in_block:
            out = self.bn3(self.linear3(out))

        if self.residual:
            if self.has_linear_in_block:
                if identity.size()[-1] != out.size()[-1]:
                    identity = self.linear0(identity)
                pass
            if identity.size()[-1] == out.size()[-1]:
                out = out + identity
            pass

        if self.has_linear_in_block:
            out = self.relu(out)

        return out

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128], global_pool=global_mean_pool, normalize=False,
                 concat=False, residual=True, position=True, gcn_num=1, has_linear_in_block=False):
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
                                                 concat=self.concat, residual=self.residual, position=self.position,
                                                 gcn_num=gcn_num, has_linear_in_block=has_linear_in_block))
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
                 concat=False, residual=True, position=True, gcn_num=1, has_linear_in_block=False):
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
                                                 concat=self.concat, residual=self.residual, position=self.position,
                                                 gcn_num=gcn_num, has_linear_in_block=has_linear_in_block))
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
        self.position_embedding_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.position_embedding_2 = nn.Linear(out_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, x):
        out = self.position_embedding_1(x)
        out = self.relu(out)
        out = self.position_embedding_2(out)
        return out

    pass


class NoAttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10, global_pool=global_max_pool):
        super().__init__()
        self.global_pool = global_pool
        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data, batched_pixel_graph):
        x = data.x
        hg = self.global_pool(x, data.batch)
        logits = self.readout_mlp(hg)
        return logits, logits

    pass


class AttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10, global_pool=global_max_pool):
        super().__init__()
        self.global_pool = global_pool
        self.attention = nn.Linear(in_dim, 1, bias=False)
        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data, batched_pixel_graph):
        x = data.x
        x_att = torch.sigmoid(self.attention(x))
        _x = (x_att * x + x) / 2
        hg = self.global_pool(_x, data.batch)
        logits = self.readout_mlp(hg)

        att_feature = self.att_feature(x_att, batched_pixel_graph)
        return logits, att_feature

    @staticmethod
    def att_feature(feature, batched_pixel_graph):
        data_where = batched_pixel_graph.data_where

        # 构造特征
        _shape = torch.max(data_where, dim=0)[0] + 1
        _size = (_shape[0], feature.shape[-1], _shape[1], _shape[2])
        _feature_for_vis = feature[batched_pixel_graph.batch]

        feature_for_vis = torch.Tensor(size=_size).to(feature.device)
        feature_for_vis[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]] = _feature_for_vis
        return feature_for_vis

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=14, normalize=True, residual=True, concat=True, pretrained=True,
                 hidden_dims1=[128, 128], hidden_dims2=[128, 128, 128, 128],
                 global_pool_1=global_mean_pool, global_pool_2=global_max_pool, gcn_num=1):
        super().__init__()

        self.model_conv = CONVNet(layer_num=conv_layer_num, pretrained=pretrained)  # 6, 13
        self.attention_class = AttentionClass(in_dim=hidden_dims2[-1], n_classes=1000, global_pool=global_pool_2)

        assert conv_layer_num == 14 or conv_layer_num == 23
        in_dim_which = -3 if conv_layer_num == 14 else -2

        self.model_gnn1 = SAGENet1(in_dim=self.model_conv.features[in_dim_which].num_features,
                                   hidden_dims=hidden_dims1, normalize=normalize, residual=residual,
                                   position=Param.position, has_linear_in_block=Param.has_linear_in_block1,
                                   concat=concat, global_pool=global_pool_1, gcn_num=gcn_num)
        self.model_gnn2 = SAGENet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=hidden_dims2,
                                   has_linear_in_block=Param.has_linear_in_block2, position=Param.position,
                                   normalize=normalize, residual=residual, concat=concat, gcn_num=gcn_num)
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
        logits, att = self.attention_class.forward(batched_graph, batched_pixel_graph)
        return logits, att

    pass


class MyDataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None):
        super(MyDataParallel, self).__init__(module, device_ids, output_device)
        self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))
        pass

    def forward(self, inputs):
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    ('Module must have its parameters and buffers on device '
                     '{} but found one of them on device {}.').format(
                         self.src_device, t.device))

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    pass


class RunnerSPE(object):

    def __init__(self):
        self.device = Param.device
        self.root_ckpt_dir = Param.root_ckpt_dir
        self.lr_s = Param.lr

        self.train_dataset = MyDataset(data_root_path=Param.data_root_path, down_ratio=Param.down_ratio, is_train=True,
                                       image_size=Param.image_size, sp_size=Param.sp_size)
        self.test_dataset = MyDataset(data_root_path=Param.data_root_path, down_ratio=Param.down_ratio, is_train=False,
                                      image_size=Param.image_size, sp_size=Param.sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=Param.batch_size, shuffle=True,
                                       num_workers=Param.num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=Param.batch_size, shuffle=False,
                                      num_workers=Param.num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=Param.conv_layer_num, normalize=Param.normalize,
                              residual=Param.residual, concat=Param.concat, hidden_dims1=Param.hidden_dims1,
                              hidden_dims2=Param.hidden_dims2, pretrained=Param.pretrained, gcn_num=Param.gcn_num,
                              global_pool_1=Param.global_pool_1, global_pool_2=Param.global_pool_2).to(self.device)

        ######################################################
        if torch.cuda.is_available():
            self.model = MyDataParallel(self.model).to(self.device)
            cudnn.benchmark = True
        ######################################################

        if Param.is_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr_s[0][1], momentum=0.9, weight_decay=Param.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr_s[0][1], weight_decay=Param.weight_decay)

        self.loss_class = nn.CrossEntropyLoss().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        pass

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)

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
        for i, (images_list, labels_list, batched_graph_list, batched_pixel_graph_list,
                images_path_list, images_small_data_list) in enumerate(self.train_loader):
            # Run
            self.optimizer.zero_grad()

            # Data
            inputs = []
            labels = torch.cat(labels_list).long().to(self.device)
            for gpu_id, (images, batched_graph, batched_pixel_graph) in enumerate(
                    zip(images_list, batched_graph_list, batched_pixel_graph_list)):
                images = images.float().to(torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                batched_graph.batch = batched_graph.batch.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                batched_graph.edge_w = batched_graph.edge_w.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                batched_graph.edge_index = batched_graph.edge_index.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                batched_pixel_graph.edge_w = batched_pixel_graph.edge_w.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(
                    torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                inputs.append([images, batched_graph, batched_pixel_graph])
                pass

            logits, _ = self.model.forward(inputs)

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
            if i % Param.train_print_freq == 0:
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
            for i, (images_list, labels_list, batched_graph_list, batched_pixel_graph_list,
                    images_path_list, images_small_data_list) in enumerate(self.test_loader):
                # Data
                inputs = []
                labels = torch.cat(labels_list).long().to(self.device)
                for gpu_id, (images, batched_graph, batched_pixel_graph) in enumerate(
                        zip(images_list, batched_graph_list, batched_pixel_graph_list)):
                    images = images.float().to(torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    batched_graph.batch = batched_graph.batch.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_graph.edge_w = batched_graph.edge_w.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_graph.edge_index = batched_graph.edge_index.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    batched_pixel_graph.batch = batched_pixel_graph.batch.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.edge_w = batched_pixel_graph.edge_w.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    inputs.append([images, batched_graph, batched_pixel_graph])
                    pass

                logits, _ = self.model.forward(inputs)
                loss = self.loss_class(logits, labels)

                # Stat
                nb_data += labels.size(0)
                epoch_test_loss += loss.detach().item()
                top_1, top_k = self._accuracy_top_k(logits, labels)
                epoch_test_acc += top_1
                epoch_test_acc_k += top_k

                # Print
                if i % Param.test_print_freq == 0:
                    Tools.print("{}-{} loss={:.4f}/{:.4f} acc={:.4f} acc5={:.4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1),
                        loss.detach().item(), epoch_test_acc/nb_data, epoch_test_acc_k/nb_data))
                    pass
                pass
            pass

        return epoch_test_loss/(len(self.test_loader)+1), epoch_test_acc/nb_data, epoch_test_acc_k/nb_data

    def visual(self, is_train=False, result_path=None, th_value=0.9, max_batch=0):
        self.model.eval()

        Tools.print()
        _loader = self.train_loader if is_train else self.test_loader
        with torch.no_grad():
            for i, (images_list, labels_list, batched_graph_list, batched_pixel_graph_list,
                    images_path_list, images_small_data_list) in enumerate(_loader):
                if 0 < max_batch < i:
                    break

                # Data
                inputs = []
                images_path_now = []
                images_small_data_now = []
                labels = torch.cat(labels_list).long().to(self.device)
                for gpu_id, (images, batched_graph, batched_pixel_graph, images_path, images_small_data) in enumerate(
                        zip(images_list, batched_graph_list, batched_pixel_graph_list,
                            images_path_list, images_small_data_list)):
                    images = images.float().to(torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    images_path_now += images_path
                    images_small_data_now += images_small_data

                    batched_graph.batch = batched_graph.batch.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_graph.edge_w = batched_graph.edge_w.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_graph.edge_index = batched_graph.edge_index.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    batched_pixel_graph.batch = batched_pixel_graph.batch.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.edge_w = batched_pixel_graph.edge_w.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))
                    batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(
                        torch.device('cuda:{}'.format(self.model.device_ids[gpu_id])))

                    inputs.append([images, batched_graph, batched_pixel_graph])
                    pass

                logits, atts = self.model.forward(inputs)

                Tools.print("{} {}".format(len(_loader), i))
                for image_path, image_small_data, att in zip(images_path_now, images_small_data_now, atts):
                    att = att[0].cpu()
                    att = att - torch.min(att)
                    att = att / torch.max(att)

                    # scale
                    att[att < th_value] = th_value
                    att = att - torch.min(att)
                    att = att / torch.max(att)

                    att = np.asarray(att * 255, dtype=np.uint8)

                    base_name = os.path.basename(image_path)
                    Image.fromarray(image_small_data).save(os.path.join(result_path, base_name))
                    Image.fromarray(att).save(os.path.join(result_path, os.path.splitext(base_name)[0] + ".bmp"))
                    pass
                pass
            pass

        pass

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
        # for file in glob.glob(root_ckpt_dir + '/*.pkl'):
        #     if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
        #         os.remove(file)
        #         pass
        #     pass
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
GCNNet-C2PC2P    0_4_4_14 3425856  2020-07-04 15:34:16 Epoch:29, Train:0.5638-0.7989/1.9539 Test:0.5556-0.7971/1.9430
GCNNet-C2PC2PC2  0_4_4_20 4344896  2020-07-08 10:20:30 Epoch:14, Train:0.5995-0.8276/1.7634 Test:0.5917-0.8264/1.7479
SAGENet-C2PC2P   1_4_4_14 5490240  2020-07-12 21:13:38 Epoch:14, Train:0.5814-0.8146/1.8494 Test:0.5748-0.8122/1.8225
SAGENet-C2PC2PC2 1_4_4_20 6442048  2020-07-15 08:19:12 Epoch:14, Train:0.6250-0.8473/1.6212 Test:0.6134-0.8424/1.6205
SAGENet-C2PC2P   1_4_4_14 10779712 2020-07-22 18:49:53 Epoch:14, Train:0.6324-0.8522/1.6052 Test:0.6128-0.8400/1.6380

2020-10-21 12:18:43 Epoch:76, Train:0.7431-0.9209/1.0510 Test:0.6974-0.8935/1.2385
"""


class Param(object):
    data_root_path = '/mnt/4T/Data/ILSVRC17/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC'
    # data_root_path = "/media/ubuntu/ALISURE-SSD/data/ImageNet/ILSVRC2015/Data/CLS-LOC"
    # data_root_path = "/media/ubuntu/ALISURE/data/ImageNet/ILSVRC2015/Data/CLS-LOC"

    pretrained = True

    # position = True
    position = False

    batch_size = 64

    image_size = 224
    train_print_freq = 100
    test_print_freq = 100
    num_workers = 40

    use_gpu = True
    device = gpu_setup(use_gpu=True, gpu_id="0")

    # has_linear_in_block = False
    has_linear_in_block1 = False
    has_linear_in_block2 = False
    # gcn_num = 1
    gcn_num = 2

    # is_sgd = False
    is_sgd = True
    if is_sgd:
        # epochs, weight_decay = 90, 5e-4
        # lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]
        epochs, weight_decay = 20, 5e-4
        lr = [[0, 0.01], [10, 0.001], [16, 0.0001]]
    else:
        epochs, weight_decay = 90, 0.0
        lr = [[0, 0.001], [30, 0.0002], [60, 0.00004]]
        pass

    # sp_size, down_ratio, conv_layer_num = 4, 4, 14  # GCNNet-C2PC2P
    sp_size, down_ratio, conv_layer_num = 4, 4, 23  # GCNNet-C2PC2PC3

    hidden_dims1 = [256, 256]
    hidden_dims2 = [512, 512, 512]

    global_pool_1, global_pool_2, pool_name = global_mean_pool, global_mean_pool, "mean_mean_pool"
    # global_pool_1, global_pool_2, pool_name = global_max_pool, global_max_pool, "max_max_pool"
    # global_pool_1, global_pool_2, pool_name = global_mean_pool, global_max_pool, "mean_max_pool"

    concat, residual, normalize = True, True, True

    _root_ckpt_dir = "./ckpt2/dgl/1_Position_PYG_CONV_Fast_ImageNet_Block/{}_{}_{}_{}_{}".format(
        is_sgd, sp_size, down_ratio, conv_layer_num, pool_name)
    root_ckpt_dir = Tools.new_dir(_root_ckpt_dir)

    @classmethod
    def print(cls):
        print(cls.__dict__)

    pass


if __name__ == '__main__':
    Param.print()

    runner = RunnerSPE()
    # runner.train(Param.epochs)

    runner.load_model("./ckpt2/dgl/1_Position_PYG_CONV_Fast_ImageNet_Block/abl/epoch_89.pkl")
    # runner.test()

    th_value = 98
    max_batch = 800
    result_path = Tools.new_dir("/mnt/4T/ALISURE/GCN/PyTorchGCN_Result/Vis_Att_{}_{}".format(th_value, max_batch))
    runner.visual(result_path=result_path, th_value=th_value / 100, max_batch=max_batch)

    pass
