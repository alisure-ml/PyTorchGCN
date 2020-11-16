import os
import cv2
import glob
import torch
import skimage
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg13_bn, vgg16_bn
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import global_mean_pool, global_max_pool


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
                                         sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        # self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
        #                                  sigma=slic_sigma, max_iter=slic_max_iter)

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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4, down_ratio=1, padding=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // down_ratio
        self.data_root_path = data_root_path

        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=padding),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
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

    def __init__(self, layer_num=6, out_dim=None, pretrained=True):  # 6, 13
        super().__init__()
        if out_dim:
            layers = [nn.Conv2d(3, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
            self.features = nn.Sequential(*layers)
        else:
            self.features = vgg13_bn(pretrained=pretrained).features[0: layer_num]
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


class MySAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super().__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        pass

    def forward(self, x, edge_index, edge_weight=None, size=None, res_n_id=None):
        if torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, None, 1, x.size(self.node_dim))
            pass
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight + x_j
        # return x_j if edge_weight is None else edge_weight * x_j
        return x_j if edge_weight is None else edge_weight * x_j + x_j

    def update(self, aggr_out, x, res_n_id):
        aggr_out = self.linear(aggr_out)
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

    pass


class MySAGEConv2(SAGEConv):

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
        out = self.relu(self.bn1(self.linear1(x))) if self.has_linear_in_block else x

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

        if self.has_linear_in_block:
            out = self.relu(out)
            out = self.bn3(self.linear3(out))

        if self.residual:
            if self.has_linear_in_block:
                if identity.size()[-1] != out.size()[-1]:
                    identity = self.linear0(identity)
                pass
            if identity.size()[-1] == out.size()[-1]:
                out = out + identity
            pass

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


class NoAttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10, global_pool=global_max_pool):
        super().__init__()
        self.global_pool = global_pool
        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data):
        x = data.x
        hg = self.global_pool(x, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class AttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10, global_pool=global_max_pool):
        super().__init__()
        self.global_pool = global_pool
        self.attention = nn.Linear(in_dim, 1, bias=False)
        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data):
        x = data.x
        x_att = torch.sigmoid(self.attention(x))
        _x = (x_att * x + x) / 2
        hg = self.global_pool(_x, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=6, normalize=True, residual=True, concat=True, pretrained=True,
                 hidden_dims1=[128, 128], hidden_dims2=[128, 128, 128, 128], has_conv=True,
                 global_pool_1=global_mean_pool, global_pool_2=global_max_pool, gcn_num=1):
        super().__init__()
        _out_dim = None if has_conv else 64
        self.model_conv = CONVNet(layer_num=conv_layer_num, out_dim=_out_dim, pretrained=pretrained)  # 6, 13
        self.attention_class = AttentionClass(in_dim=hidden_dims2[-1], n_classes=10, global_pool=global_pool_2)

        self.model_gnn1 = SAGENet1(in_dim=self.model_conv.features[-2].num_features, hidden_dims=hidden_dims1,
                                   normalize=normalize, residual=residual, position=Param.position,
                                   has_linear_in_block=Param.has_linear_in_block1,
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
        logits = self.attention_class.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self):
        self.device = Param.device
        self.root_ckpt_dir = Param.root_ckpt_dir
        self.lr_s = Param.lr

        self.train_dataset = MyDataset(data_root_path=Param.data_root_path, down_ratio=Param.down_ratio, is_train=True,
                                       image_size=Param.image_size, sp_size=Param.sp_size, padding=Param.padding)
        self.test_dataset = MyDataset(data_root_path=Param.data_root_path, down_ratio=Param.down_ratio, is_train=False,
                                      image_size=Param.image_size, sp_size=Param.sp_size, padding=Param.padding)

        self.train_loader = DataLoader(self.train_dataset, batch_size=Param.batch_size, shuffle=True,
                                       num_workers=Param.num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=Param.batch_size, shuffle=False,
                                      num_workers=Param.num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=Param.conv_layer_num, normalize=Param.normalize, has_conv=Param.has_conv,
                              residual=Param.residual, concat=Param.concat, hidden_dims1=Param.hidden_dims1,
                              hidden_dims2=Param.hidden_dims2, pretrained=Param.pretrained, gcn_num=Param.gcn_num,
                              global_pool_1=Param.global_pool_1, global_pool_2=Param.global_pool_2).to(self.device)

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
                if i % Param.test_print_freq == 0:
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
4_1_6      2020-08-29 02:04:33 Epoch:145, Train:0.9594-0.9998/0.1223 Test:0.8812-0.9965/0.3735
4_1_6  Att 2020-08-29 01:45:03 Epoch:141, Train:0.9653-0.9998/0.1100 Test:0.8860-0.9959/0.3721
2_2_13     2020-09-22 19:02:36 Epoch:134, Train:0.9751-0.9998/0.0754 Test:0.9063-0.9965/0.3305
2_2_13 Att 2020-09-22 19:02:36 Epoch:111, Train:0.9684-0.9998/0.0950 Test:0.9089-0.9967/0.3060

4_1_6  Att + SGD 2020-09-23 01:27:14 Epoch:133, Train:0.9652-0.9998/0.1111 Test:0.8854-0.9956/0.3725
4_1_6  Att * ReLU SGD 2Linear :58:33 Epoch:138, Train:0.9713-0.9999/0.0922 Test:0.8944-0.9962/0.3491

2_2_13 Att + SGD 2020-09-22 19:02:36 Epoch:144, Train:0.9829-0.9999/0.0562 Test:0.9039-0.9972/0.3407
2_2_13 Att + ReLU SGD 09-23 07:32:00 Epoch:105, Train:0.9648-0.9997/0.1084 Test:0.9069-0.9968/0.3037
2_2_13 Att + ReLU Adam 9-23 05:28:13 Epoch: 91, Train:0.9927-1.0000/0.0225 Test:0.9004-0.9966/0.5217
2_2_13 Att * ReLU SGD 2Linear :08:25 Epoch:149, Train:0.9859-1.0000/0.0481 Test:0.9121-0.9967/0.3270

2_2_13 Att * ReLU Adam 2Linear mean_max_pool  594112 lr:0.001 Train:0.9972-1.0000/0.0099 Test:0.9178-0.9975/0.4073
2_2_13 Att * ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 Train:0.9947-1.0000/0.0208 Test:0.9147-0.9979/0.3242
2_2_13 Att * ReLU  SGD 2Linear mean_mean_pool 594112  lr:0.01 Train:0.9933-1.0000/0.0266 Test:0.9056-0.9965/0.3556

2_2_13 Att * ReLU Adam 2Linear mean_mean_pool 594112 lr:0.001 wd:0.0005 Train:0.9772-0.9999/0.0718 Test:0.9010-0.9962/0.3478

4_1_6  Att * ReLU Adam 2Linear mean_max_pool  331008 lr:0.001 Train:0.9823-1.0000/0.0506 Test:0.9028-0.9972/0.3696
4_1_6  Att * ReLU  SGD 2Linear mean_max_pool  331008  lr:0.01 Train:0.9813-1.0000/0.0619 Test:0.9053-0.9973/0.3177
4_1_6  Att * ReLU  SGD 2Linear mean_max_pool  331008   lr:0.1 Train:0.8799-0.9966/0.3479 Test:0.8564-0.9947/0.4270

4_1_6  Att *+ ReLU Adam 2Linear mean_max_pool  331008 lr:0.001 Train:0.9842-1.0000/0.0475 Test:0.9033-0.9973/0.3636
4_1_6  Att *+ ReLU  SGD 2Linear mean_max_pool  331008  lr:0.01 Train:0.9852-1.0000/0.0528 Test:0.9061-0.9970/0.3159
4_1_6  Att *+ ReLU  SGD 2Linear mean_max_pool  331008  lr:0.01 padding:4 Epoch:115, Train:0.9792-1.0000/0.0667 Test:0.9050-0.9968/0.3188

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:512 Epoch:54, Train:0.9841-1.0000/0.0534 Test:0.8989-0.9958/0.3263
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.02 padding:2 bs:512 Epoch:97, Train:0.9964-1.0000/0.0139 Test:0.9139-0.9974/0.3748
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 Epoch:143, Train:0.9958-1.0000/0.0184 Test:0.9202-0.9972/0.3086

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  1072960  lr:0.01 padding:2 bs:64 256*4 Epoch:146, Train:0.9987-1.0000/0.0089 Test:0.9294-0.9971/0.2792
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  3326400  lr:0.01 padding:2 bs:64 256*2 512*4 Epoch:134, Train:0.9995-1.0000/0.0061 Test:0.9328-0.9972/0.2600

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  1798720  lr:0.01 padding:2 bs:64 128*2 256*2 512*2 Epoch:137, Train:0.9993-1.0000/0.0072 Test:0.9300-0.9973/0.2687
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  1798720  lr:0.01 padding:4 bs:64 128*2 256*2 512*2 Epoch:146, Train:0.9980-1.0000/0.0103 Test:0.9302-0.9982/0.2616
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  1898304  lr:0.01 padding:2 bs:64 128*2 128*2 256*2 512*2 Epoch:72, Train:0.9944-1.0000/0.0228 Test:0.9261-0.9972/0.2696
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  2835392  lr:0.01 padding:2 bs:64 128*3 256*3 512*3 Epoch:136, Train:0.9990-1.0000/0.0069 Test:0.9281-0.9980/0.2861

2_2_13  NoAtt *+ ReLU  SGD 2Linear mean_max_pool 3325888  lr:0.01 padding:2 bs:64 256*2 512*4 Epoch:124, Train:0.9990-1.0000/0.0075 Test:0.9269-0.9981/0.2830
2_2_13  NoAtt *+ ReLU  SGD 2Linear mean_max_pool  593984  lr:0.01 padding:2 bs:64 128*2 128*4 Epoch:114, Train:0.9943-1.0000/0.0233 Test:0.9182-0.9967/0.3098

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 128*2 128*4 NoPre Epoch:106, Train:0.9955-1.0000/0.0192 Test:0.9146-0.9969/0.3300
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool 3326400  lr:0.01 padding:2 bs:64 256*2 512*4 NoPre Epoch:124, Train:0.9995-1.0000/0.0062 Test:0.9252-0.9978/0.2830

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.0256 padding:2 bs:64 Epoch:111, Train:0.9932-1.0000/0.0258 Test:0.9210-0.9978/0.2913
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.0256 padding:2 bs:512 Epoch:71, Train:0.9913-1.0000/0.0278 Test:0.9112-0.9965/0.3341
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 PosHasReLU Epoch:107, Train:0.9944-1.0000/0.0225 Test:0.9163-0.9967/0.3142

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 NoPos Epoch:116, Train:0.9941-1.0000/0.0232 Test:0.9200-0.9966/0.3091
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 PosSigmoid :129, Train:0.9948-1.0000/0.0216 Test:0.9189-0.9972/0.3162

2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 ignore:0.0 Epoch:146, Train:0.9961-1.0000/0.0179 Test:0.9211-0.9968/0.3160
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 ignore:0.1 Epoch:109, Train:0.9924-1.0000/0.0278 Test:0.9172-0.9964/0.3130
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 ignore:0.2 Epoch:111, Train:0.9921-1.0000/0.0300 Test:0.9154-0.9970/0.3134
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 ignore:0.3 Epoch:107, Train:0.9889-1.0000/0.0369 Test:0.9143-0.9962/0.3193

"""


"""
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 3326400 lr:0.01 padding:2 bs:64   Block 256*2 512*4 Epoch:134, Train:0.9992-1.0000/0.0057 Test:0.9294-0.9975/0.3042
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793536 lr:0.01 padding:2 bs:64   Block 128*2 128*4 Epoch:142, Train:0.9960-1.0000/0.0167 Test:0.9197-0.9968/0.3336
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  593856 lr:0.01 padding:2 bs:64 NoBlock 128*2 128*4 Epoch:149, Train:0.9960-1.0000/0.0181 Test:0.9199-0.9974/0.3082


2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  593856 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 128*2 128*4 Epoch:121, Train:0.9960-1.0000/0.0187 Test:0.9183-0.9970/0.3140
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793536 lr:0.01 padding:2 bs:64   Block gcn_num:1 128*2 128*4 Epoch:125, Train:0.9948-1.0000/0.0193 Test:0.9197-0.9970/0.3312
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  892608 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 128*2 128*4 Epoch:115, Train:0.9973-1.0000/0.0134 Test:0.9221-0.9973/0.3117
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 1092288 lr:0.01 padding:2 bs:64   Block gcn_num:2 128*2 128*4 Epoch:124, Train:0.9951-1.0000/0.0182 Test:0.9192-0.9975/0.3269

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  593856 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 128*2 128*4 Epoch:132, Train:0.9953-1.0000/0.0187 Test:0.9209-0.9967/0.3070
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793536 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 128*2 128*4 Epoch:134, Train:0.9974-1.0000/0.0124 Test:0.9235-0.9976/0.3077

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793024 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 128*2 128*3 Epoch:146, Train:0.9971-1.0000/0.0137 Test:0.9251-0.9976/0.2969
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 6877760 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 512*4 Epoch:136, Train:0.9995-1.0000/0.0044 Test:0.9300-0.9976/0.2879

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 2209600 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Epoch:110, Train:0.9990-1.0000/0.0074 Test:0.9331-0.9982/0.2593
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 2605376 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*4 Epoch:76, Train:0.9970-1.0000/0.0154 Test:0.9307-0.9969/0.2590

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793024 lr:0.0256 padding:2 bs:64 NoBlock gcn_num:2 128*2 128*3 Epoch:138, Train:0.9971-1.0000/0.0131 Test:0.9237-0.9976/0.3016

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 1598528 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 NoPos Epoch:115, Train:0.9989-1.0000/0.0084 Test:0.9283-0.9978/0.2730
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 5299776 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 512*3 Epoch:126, Train:0.9997-1.0000/0.0044 Test:0.9311-0.9974/0.2708
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  693440 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 128*1 128*3 Epoch:137, Train:0.9973-1.0000/0.0134 Test:0.9236-0.9973/0.3115

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool  793024 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 128*2 128*3 Epoch:126, Train:0.9975-1.0000/0.0136 Test:0.9273-0.9978/0.2879
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 5299776 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 512*3 Epoch:124, Train:0.9995-1.0000/0.0044 Test:0.9302-0.9971/0.2871

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 1418048 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 256*2 256*4 Epoch:104, Train:0.9980-1.0000/0.0122 Test:0.9303-0.9978/0.2619
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 3326016 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 256*2 512*4 Epoch:130, Train:0.9994-1.0000/0.0066 Test:0.9306-0.9979/0.2516

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 1418432 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 256*2 256*4 bias Epoch:121, Train:0.9984-1.0000/0.0108 Test:0.9273-0.9977/0.2668
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 3326400 lr:0.01 padding:2 bs:64 NoBlock gcn_num:1 256*2 512*4 bias Epoch:148, Train:0.9996-1.0000/0.0056 Test:0.9314-0.9982/0.2616

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 5410112 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 NoPos Epoch:142, Train:0.9997-1.0000/0.0047 Test:0.9315-0.9983/0.2504
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 7795264 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 Epoch:136, Train:0.9999-1.0000/0.0039 Test:0.9364-0.9982/0.2315
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 8484160 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 embedding Epoch:134, Train:0.9998-1.0000/0.0037 Test:0.9359-0.9977/0.2344
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 2340928 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 embedding Epoch:107, Train:0.9992-1.0000/0.0069 Test:0.9308-0.9977/0.2660

2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 8484160 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 embedding 90 Epoch:74, Train:0.9989-1.0000/0.0099 Test:0.9335-0.9976/0.2387
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 2340928 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 embedding 90 Epoch:64, Train:0.9956-1.0000/0.0205 Test:0.9273-0.9978/0.2623
"""


"""
2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 8484160 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 Emd 300 Epoch:155, Train:1.0000-1.0000/0.0029 Test:0.9393-0.9984/0.2156
4_1_6  Att *+ ReLU SGD 2Linear mean_max_pool 5717568 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 512*3 Emd 300 Epoch:228, Train:1.0000-1.0000/0.0027 Test:0.9325-0.9980/0.2471
4_1_6  Att *+ ReLU SGD 2Linear mean_max_pool 2102592 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:203, Train:0.9997-1.0000/0.0052 Test:0.9322-0.9981/0.2510
4_1_3  Att *+ ReLU SGD 2Linear mean_max_pool 2065536 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:214, Train:0.9996-1.0000/0.0055 Test:0.9192-0.9972/0.3069
4_1_0  Att *+ ReLU SGD 2Linear mean_max_pool 2064000 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:176, Train:0.9949-1.0000/0.0251 Test:0.8713-0.9944/0.4637

MY 4_1_3  Att *+ ReLU SGD 2Linear mean_max_pool 1410176 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:242, Train:0.9981-1.0000/0.0130 Test:0.8562-0.9926/0.5487
MY 4_1_6  Att *+ ReLU SGD 2Linear mean_max_pool 1447232 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:113, Train:0.9943-1.0000/0.0289 Test:0.9044-0.9978/0.3116
MY 2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 1685568 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 256*2 256*3 Emd 300 Epoch:167, Train:0.9989-1.0000/0.0079 Test:0.9296-0.9978/0.2768
MY 2_2_13 Att *+ ReLU SGD 2Linear mean_max_pool 5862720 lr:0.01 padding:2 bs:64 NoBlock gcn_num:2 512*2 512*3 Emd 300 Epoch:167, Train:0.9998-1.0000/0.0038 Test:0.9339-0.9977/0.2506
"""


class Param(object):
    # data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    # data_root_path = '/home/ubuntu/ALISURE/data/cifar'
    data_root_path = "/media/ubuntu/4T/ALISURE/Data/cifar"

    pretrained = True

    position = True
    # position = False

    batch_size = 64
    # batch_size = 512

    image_size = 32
    train_print_freq = 100
    test_print_freq = 50
    num_workers = 16

    # sp_size, down_ratio, conv_layer_num = 4, 1, 3
    # sp_size, down_ratio, conv_layer_num = 4, 1, 6
    sp_size, down_ratio, conv_layer_num = 2, 2, 13

    use_gpu = True
    # device = gpu_setup(use_gpu=True, gpu_id="0")
    # device = gpu_setup(use_gpu=True, gpu_id="1")
    device = gpu_setup(use_gpu=True, gpu_id="2")
    # device = gpu_setup(use_gpu=True, gpu_id="3")

    padding = 2
    # padding = 4

    has_conv = True
    # has_conv = False

    # has_linear_in_block = False
    has_linear_in_block1 = False
    has_linear_in_block2 = False
    # gcn_num = 1
    gcn_num = 2

    # is_sgd = False
    is_sgd = True
    if is_sgd:
        # epochs, weight_decay = 150, 5e-4
        # lr = [[0, 0.1], [50, 0.01], [100, 0.001], [130, 0.0001]]
        # lr = [[0, 0.01], [50, 0.001], [100, 0.0001]]
        # lr = [[0, 0.0256], [50, 0.00256], [100, 0.000256]]

        # epochs, weight_decay = 90, 5e-4
        # lr = [[0, 0.01], [30, 0.001], [60, 0.0001]]

        epochs, weight_decay = 300, 5e-4
        lr = [[0, 0.01], [100, 0.001], [200, 0.0001]]
    else:
        epochs, weight_decay, lr = 100, 0.0, [[0, 0.001], [50, 0.0002], [75, 0.00004]]
        pass

    # hidden_dims1 = [128, 128]
    # hidden_dims2 = [128, 128, 128, 128]
    # hidden_dims1 = [128, 128]
    # hidden_dims2 = [128, 128, 128]
    # hidden_dims1 = [256, 256]
    # hidden_dims2 = [512, 512, 512, 512]
    # hidden_dims1 = [256, 256]
    # hidden_dims2 = [256, 256, 256]
    # hidden_dims1 = [256, 256]
    # hidden_dims2 = [256, 256, 256, 256]
    # hidden_dims1 = [256, 256]
    # hidden_dims2 = [512, 512, 512]
    hidden_dims1 = [512, 512]
    hidden_dims2 = [512, 512, 512]

    # global_pool_1, global_pool_2, pool_name = global_mean_pool, global_mean_pool, "mean_mean_pool"
    # global_pool_1, global_pool_2, pool_name = global_max_pool, global_max_pool, "max_max_pool"
    global_pool_1, global_pool_2, pool_name = global_mean_pool, global_max_pool, "mean_max_pool"

    concat, residual, normalize = True, True, True

    _root_ckpt_dir = "./ckpt2/dgl/1_Position_PYG_CONV_Fast_CIFAR10_Block/{}_{}_{}_{}_{}".format(
        is_sgd, sp_size, down_ratio, conv_layer_num, pool_name)
    root_ckpt_dir = Tools.new_dir(_root_ckpt_dir)

    @classmethod
    def print(cls):
        print(cls.__dict__)

    pass


if __name__ == '__main__':
    Param.print()

    runner = RunnerSPE()
    runner.train(Param.epochs)

    pass
