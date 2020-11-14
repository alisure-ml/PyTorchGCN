import os
import cv2
import glob
import torch
import skimage
import argparse
import platform
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

        try:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        except Exception:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter)
            pass

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

    def __init__(self, c=0, pretrained=True):  # 6, 13
        super().__init__()
        self.c = c
        if self.c == 0:
            layer_num = 0
        elif self.c == 1:
            layer_num = 3
        elif self.c == 2:
            layer_num = 6
        elif self.c == 4:
            layer_num = 13
        else:
            raise Exception("c={}".format(self.c))

        self.features = vgg13_bn(pretrained=pretrained).features[0: layer_num]

        self.out_dim = 3 if self.c == 0 else self.features[-2].num_features
        pass

    def forward(self, x):
        e = self.features(x) if self.c > 0 else x
        return e

    pass


class MySAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels, concat=True, bias=True, **kwargs):
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


class MyMLPGNN(MessagePassing):

    def __init__(self, in_channels, out_channels, concat=True, bias=True, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        pass

    def forward(self, x, edge_index, edge_weight=None, size=None, res_n_id=None):
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return x_j

    def update(self, aggr_out, x, res_n_id):
        aggr_out = self.linear(x)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

    pass


class MySAGEConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, position=True, concat=True, residual=True, gcn_num=1, gnn_layer=MySAGEConv):
        super().__init__()
        self.position = position
        self.residual = residual
        self.gcn_num = gcn_num

        _in_dim = in_dim
        self.gcn = gnn_layer(_in_dim, out_dim, concat=concat)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if self.gcn_num == 2:
            self.gcn2 = gnn_layer(out_dim, out_dim, concat=concat)
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
        out = self.relu(self.bn2(out))

        if self.gcn_num == 2:
            position_embedding = self.pos2(data.edge_w) if self.position else None
            out = self.gcn2(out, data.edge_index, edge_weight=position_embedding)
            out = self.relu(self.bn22(out))
        ##################################

        if self.residual:
            if identity.size()[-1] == out.size()[-1]:
                out = out + identity
            pass

        return out

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128], global_pool=global_mean_pool,
                 concat=True, residual=True, position=True, gcn_num=1, gnn_layer=MySAGEConv, is_mean=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.position = position
        self.is_mean = is_mean

        embedding_dim = self.hidden_dims[0]
        self.embedding_h = nn.Linear(self.in_dim, embedding_dim, bias=False)

        if not self.is_mean:
            self.gcn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.gcn_list.append(MySAGEConvBlock(embedding_dim, hidden_dim, concat=concat, gnn_layer=gnn_layer,
                                                     residual=self.residual, position=self.position, gcn_num=gcn_num))
                embedding_dim = hidden_dim
                pass
            pass

        self.global_pool = global_pool
        self.out_dim = embedding_dim
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        if not self.is_mean:
            for gcn in self.gcn_list:
                hidden_nodes_feat = gcn(hidden_nodes_feat, data)
                pass
            pass

        hg = self.global_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class SAGENet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128], concat=True,
                 residual=True, position=True, gcn_num=1, gnn_layer=MySAGEConv, is_mean=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.position = position
        self.is_mean = is_mean

        embedding_dim = self.hidden_dims[0]
        self.embedding_h = nn.Linear(self.in_dim, embedding_dim, bias=False)

        if not self.is_mean:
            self.gcn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.gcn_list.append(MySAGEConvBlock(embedding_dim, hidden_dim, concat=concat, gnn_layer=gnn_layer,
                                                     residual=self.residual, position=self.position, gcn_num=gcn_num))
                embedding_dim = hidden_dim
                pass
            pass

        self.out_dim = embedding_dim
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        if not self.is_mean:
            for gcn in self.gcn_list:
                hidden_nodes_feat = gcn(hidden_nodes_feat, data)
                pass
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

    def __init__(self):
        super().__init__()
        self.pl_cnn = CONVNet(c=param.c, pretrained=param.pretrained)  # 6, 13

        self.sl_gnn = SAGENet1(in_dim=self.pl_cnn.out_dim, hidden_dims=param.hidden_dims1, concat=param.concat,
                               residual=param.residual, gnn_layer=param.gnn_layer_1, is_mean=param.is_mean_1,
                               position=param.position, global_pool=param.global_pool_1, gcn_num=param.gcn_num)

        self.il_gnn = SAGENet2(in_dim=self.sl_gnn.out_dim, hidden_dims=param.hidden_dims2,
                               gnn_layer=param.gnn_layer_2, concat=param.concat, is_mean=param.is_mean_2,
                               residual=param.residual, position=param.position, gcn_num=param.gcn_num)

        self.alc = AttentionClass(in_dim=self.il_gnn.out_dim, n_classes=10,
                                  global_pool=param.global_pool_2, is_attention=param.is_attention)
        pass

    def forward(self, images, batched_graph, batched_pixel_graph):
        # model 1
        conv_feature = self.pl_cnn(images)

        # model 2
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = conv_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.sl_gnn.forward(batched_pixel_graph)

        # model 3
        batched_graph.x = gcn1_feature
        gcn2_feature = self.il_gnn.forward(batched_graph)

        batched_graph.x = gcn2_feature
        logits = self.alc.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self):
        self.device = param.device
        self.root_ckpt_dir = param.root_ckpt_dir
        self.lr_s = param.lr

        self.train_dataset = MyDataset(data_root_path=param.data_root, down_ratio=param.down_ratio, is_train=True,
                                       image_size=param.image_size, sp_size=param.sp_size, padding=param.padding)
        self.test_dataset = MyDataset(data_root_path=param.data_root, down_ratio=param.down_ratio, is_train=False,
                                      image_size=param.image_size, sp_size=param.sp_size, padding=param.padding)

        self.train_loader = DataLoader(self.train_dataset, batch_size=param.batch_size, shuffle=True,
                                       num_workers=param.num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=param.batch_size, shuffle=False,
                                      num_workers=param.num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet().to(self.device)

        if param.is_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr_s[0][1], momentum=0.9, weight_decay=param.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr_s[0][1], weight_decay=param.weight_decay)

        self.loss_class = nn.CrossEntropyLoss().to(self.device)

        param_num = "Total param: {}".format(self._view_model_param(self.model))
        Tools.print(param_num)
        Tools.write_to_txt(param.log_path, param_num + "\n")
        pass

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)

        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs, start_epoch=0):
        max_test_acc = 0.0
        for epoch in range(start_epoch, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            epoch_loss, epoch_train_acc, epoch_train_acc_k = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            test_loss, epoch_test_acc, epoch_test_acc_k = self.test()

            result_str = 'Epoch:{:02d}, Train:{:.4f}-{:.4f}/{:.4f} Test:{:.4f}-{:.4f}/{:.4f}'.format(
                epoch, epoch_train_acc, epoch_train_acc_k, epoch_loss, epoch_test_acc, epoch_test_acc_k, test_loss)
            Tools.print(result_str)

            if epoch_test_acc > max_test_acc:
                max_test_acc = epoch_test_acc
                Tools.write_to_txt(param.log_path, result_str + "\n")
                pass

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
            if i % param.train_print_freq == 0:
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
                if i % param.test_print_freq == 0:
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


class Param(object):

    def __init__(self, arc=None):
        if arc is None:
            self.arc = {
                "name": "final",
                "gpu_id": 0,
                "PL": 4,
                "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
                "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
                "LC": {"name": "ALC", "pool": "max"}
            }
        else:
            self.arc = arc
            pass

        # PL-CNN
        self.c = self.arc["PL"]

        # SL-GNN
        self.gcn_num = self.arc["SL"]["gcn_num"]
        self.is_mean_1 = False if self.arc["SL"]["mean"] == 0 else True
        self.gnn_layer_1 = MySAGEConv if self.arc["SL"]["gcn_type"] == 1 else MyMLPGNN
        self.hidden_dims1 = [self.arc["SL"]["dim"]] * self.arc["SL"]["num"]
        self.global_pool_1 = global_max_pool if self.arc["SL"]["pool"] == "max" else global_mean_pool
        if self.arc["SL"]["name"] == "BG":
            self.position, self.residual = False, False
        elif self.arc["SL"]["name"] == "PG":
            self.position, self.residual = True, False
        elif self.arc["SL"]["name"] == "BRG":
            self.position, self.residual = False, True
        elif self.arc["SL"]["name"] == "PRG":
            self.position, self.residual = True, True
            pass

        # IL-GNN
        self.is_mean_2 = False if self.arc["IL"]["mean"] == 0 else True
        self.gnn_layer_2 = MySAGEConv if self.arc["IL"]["gcn_type"] == 1 else MyMLPGNN
        self.hidden_dims2 = [self.arc["IL"]["dim"]] * self.arc["IL"]["num"]
        if self.arc["IL"]["name"] == "BG":
            self.position, self.residual = False, False
        elif self.arc["IL"]["name"] == "PG":
            self.position, self.residual = True, False
        elif self.arc["IL"]["name"] == "BRG":
            self.position, self.residual = False, True
        elif self.arc["IL"]["name"] == "PRG":
            self.position, self.residual = True, True
            pass

        # ALC
        self.global_pool_2 = global_max_pool if self.arc["LC"]["pool"] == "max" else global_mean_pool
        self.is_attention = True if self.arc["LC"]["name"] == "ALC" else False

        self.batch_size = 64
        self.image_size = 32
        # self.padding = 2
        self.padding = 4
        self.concat = True

        self.pretrained = True

        self.train_print_freq = 400
        self.test_print_freq = 100
        self.num_workers = 10

        if self.c == 0 or self.c == 1 or self.c == 2:
            self.sp_size, self.down_ratio = 4, 1
        else:
            self.sp_size, self.down_ratio = 2, 2
            pass

        # self.is_sgd = False
        self.is_sgd = True
        self.epochs, self.weight_decay, self.lr = self.get_optim(self.is_sgd)
        self.device = gpu_setup(use_gpu=True, gpu_id=str(self.arc["gpu_id"]))
        self.data_root = self.get_data_root()

        self.name = self.get_name()
        self.root_ckpt_dir = Tools.new_dir("./ckpt3/dgl/Abl_CIFAR10/{}".format(self.name))
        self.log_path = Tools.new_dir("./ckpt3/log/Abl_CIFAR10/{}.txt".format(self.name))
        pass

    @staticmethod
    def get_optim(is_sgd):
        if is_sgd:
            epochs, weight_decay = 180, 5e-4
            lr = [[0, 0.01], [80, 0.001], [140, 0.0001]]
        else:
            epochs, weight_decay, lr = 100, 0.0, [[0, 0.001], [50, 0.0002], [75, 0.00004]]
            pass
        return epochs, weight_decay, lr

    def get_name(self):
        name = "{}_{}_{}_{}_{}_{}_{}_{}_PL-{}_SL-{}_{}_{}_{}_{}_{}_{}_IL-{}_{}_{}_{}_{}_{}_LC-{}_{}".format(
            self.arc["name"], self.arc["gpu_id"],
            self.image_size, self.padding, self.epochs, self.batch_size, self.sp_size, self.down_ratio,
            self.arc["PL"],
            self.arc["SL"]["name"], self.arc["SL"]["num"], self.arc["SL"]["dim"],
            self.arc["SL"]["gcn_type"], self.arc["SL"]["gcn_num"], self.arc["SL"]["mean"], self.arc["SL"]["pool"],
            self.arc["IL"]["name"], self.arc["IL"]["num"], self.arc["IL"]["dim"],
            self.arc["IL"]["gcn_type"], self.arc["IL"]["gcn_num"], self.arc["IL"]["mean"],
            self.arc["LC"]["name"], self.arc["LC"]["pool"])
        return name

    def print(self):
        print(self.name)
        print(self.__dict__)
        Tools.write_to_txt(self.log_path, self.name + "\n")
        pass

    @staticmethod
    def get_data_root():
        if "Linux" in platform.platform():
            data_root_path_list = ['/mnt/4T/Data/cifar/cifar-10', '/home/ubuntu/ALISURE/data/cifar',
                                   "/mnt/4T/Data/data/CIFAR", '/media/ubuntu/4T/ALISURE/Data/cifar',
                                   "/media/ubuntu/data1/ALISURE/cifar10", "/mnt/4T/ALISURE/cifar10"]
            data_root = None
            for data_root_path in data_root_path_list:
                if os.path.isdir(data_root_path):
                    data_root = data_root_path
                    break
                pass
            pass
        else:
            data_root = "F:\\data\\cifar10"

        if data_root is None:
            raise Exception("data root is {}".format(data_root))
        return data_root

    pass


class AblConfig(object):

    @classmethod
    def get_config(cls, t, gpu_id):
        Tools.print("{} {}".format(t, gpu_id))
        if t == "1_a_1":
            return cls.get_table_1_a_1(name=t, gpu_id=gpu_id)
        if t == "1_a_3":
            return cls.get_table_1_a_3(name=t, gpu_id=gpu_id)
        if t == "1_b_1":
            return cls.get_table_1_b_1(name=t, gpu_id=gpu_id)
        if t == "1_b_2":
            return cls.get_table_1_b_2(name=t, gpu_id=gpu_id)
        if t == "1_b_3":
            return cls.get_table_1_b_3(name=t, gpu_id=gpu_id)
        if t == "2_a":
            return cls.get_table_2_a(name=t, gpu_id=gpu_id)
        if t == "2_b":
            return cls.get_table_2_b(name=t, gpu_id=gpu_id)
        if t == "3":
            return cls.get_table_3(name=t, gpu_id=gpu_id)
        if t == "5":
            return cls.get_table_5(name=t, gpu_id=gpu_id)
        pass

    # Table 1 (a): superpixel representation method

    @staticmethod
    def get_table_1_a_1(name, gpu_id=0):

        # C0 Mean
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 1, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C2 Mean
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 1, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C4 Mean
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 1, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03

    @staticmethod
    def get_table_1_a_3(name, gpu_id=0):
        # C0 GNN
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C2 GNN
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C4 GNN
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03

    # Table 1 (b): basic layers in sl-gnn and il-gnn

    @staticmethod
    def get_table_1_b_1(name, gpu_id=0):
        # C0 MLP
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 2, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 2, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C2 MLP
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 2, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 2, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C4 MLP
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 2, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 2, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03

    @staticmethod
    def get_table_1_b_2(name, gpu_id=0):
        # C0 GraphSAGE
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C2 GraphSAGE
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C4 GraphSAGE
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03

    @staticmethod
    def get_table_1_b_3(name, gpu_id=0):
        # C0 PGraphSAGE
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "PG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C2 PGraphSAGE
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "PG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # C4 PGraphSAGE
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PG", "num": 1, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PG", "num": 2, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03

    # Table 2 (a): gnn blocks in sl-gnn and il-gnn

    @staticmethod
    def get_table_2_a(name, gpu_id=0):
        # BG
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # PG
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # BRG
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "BRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "BRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # PRG
        arc_04 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }

        return arc_01, arc_02, arc_03, arc_04

    # Table 2 (b): readout in sl-gnn and linear classifier

    @staticmethod
    def get_table_2_b(name, gpu_id=0):
        # avg, avg
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # avg, max
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "max"}
        }
        # max, avg
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "max"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "avg"}
        }
        # max, max
        arc_04 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "max"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "max"}
        }

        return arc_01, arc_02, arc_03, arc_04

    # Table 3: the importance of each module

    @staticmethod
    def get_table_3(name, gpu_id=0):

        # C0 PRG2 PRG3 ALC
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }
        # C4 Linear PRG3 ALC
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 1, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }
        # C4 PRG2 Linear ALC
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 1},
            "LC": {"name": "ALC", "pool": "max"}
        }
        # C4 PRG2 PRG3 LC
        arc_04 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "LC", "pool": "max"}
        }
        # C4 PRG2 PRG3 ALC
        arc_05 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }

        return arc_01, arc_02, arc_03, arc_04, arc_05

    # Table 5: comparisons of different methods

    @staticmethod
    def get_table_5(name, gpu_id=0):
        # C0 PRG2 PRG3 ALC
        arc_01 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 0,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }
        # C2 PRG2 PRG3 ALC
        arc_02 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 2,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }
        # C4 PRG2 PRG3 ALC
        arc_03 = {
            "name": name,
            "gpu_id": gpu_id, "PL": 4,
            "SL": {"name": "PRG", "num": 2, "dim": 128, "gcn_type": 1, "gcn_num": 2, "mean": 0, "pool": "avg"},
            "IL": {"name": "PRG", "num": 3, "dim": 256, "gcn_type": 1, "gcn_num": 2, "mean": 0},
            "LC": {"name": "ALC", "pool": "max"}
        }

        return arc_01, arc_02, arc_03

    pass


if __name__ == '__main__':

    """
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 1_a_1 --g 0
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 1_a_3 --g 1
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 1_b_1 --g 2
    
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 1_b_2 --g 0
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 1_b_3 --g 1
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 2_a --g 2
    
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 2_b --g 0
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 3 --g 1
    python SPERunner_1_Position_PYG_CONV_Fast_CIFAR10_Block_Abl.py --t 5 --g 2
    """

    arg = argparse.ArgumentParser()
    arg.add_argument("--g", required=True, type=int, default=1, help="gpu id")
    arg.add_argument("--t", required=True, type=str, default="1_a_1", help="table id")
    args = arg.parse_args()

    config_list = AblConfig.get_config(t=args.t, gpu_id=args.g)
    for config in config_list:
        param = Param(arc=config)
        param.print()

        runner = RunnerSPE()
        runner.train(param.epochs)
        pass

    pass
