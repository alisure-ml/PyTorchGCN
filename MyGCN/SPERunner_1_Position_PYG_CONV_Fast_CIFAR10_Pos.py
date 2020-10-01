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
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform


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


class MySAGEConv2(SAGEConv):

    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, normalize=normalize, concat=concat, bias=bias, **kwargs)
        pass

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight + x_j
        # return x_j if edge_weight is None else edge_weight * x_j
        return x_j if edge_weight is None else edge_weight * x_j + x_j

    pass


class MySAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super(MySAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat

        self.linear1 = nn.Linear(2 * self.in_channels if concat else self.in_channels, self.out_channels, bias=bias)
        self.linear2 = nn.Linear(self.in_channels, self.out_channels, bias=False)
        pass

    def forward(self, x, edge_index, edge_weight=None, size=None, res_n_id=None):
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else self.linear2(edge_weight + x_j)
        # return x_j if edge_weight is None else self.linear(torch.cat([x_j, edge_weight], dim=-1))

    def update(self, aggr_out, x, res_n_id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        aggr_out = self.linear1(aggr_out)
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=64, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, concat=False, position=True):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.concat = concat
        self.position = position

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(MySAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        if self.position:
            self.position_list = nn.ModuleList()
            self.position_list.append(PositionEmbedding(2, in_dim))
            for hidden_dim in self.hidden_dims[:-1]:
                self.position_list.append(PositionEmbedding(2, hidden_dim))
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for i, (gcn, bn) in enumerate(zip(self.gcn_list, self.bn_list)):
            h_in = hidden_nodes_feat

            position_embedding = self.position_list[i](data.edge_w) if self.position else None
            hidden_nodes_feat = gcn(h_in, data.edge_index, edge_weight=position_embedding)

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

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, concat=False, position=True):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat
        self.position = position

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(MySAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        if self.position:
            self.position_list = nn.ModuleList()
            self.position_list.append(PositionEmbedding(2, in_dim))
            for hidden_dim in self.hidden_dims[:-1]:
                self.position_list.append(PositionEmbedding(2, hidden_dim))
            pass
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for i, (gcn, bn) in enumerate(zip(self.gcn_list, self.bn_list)):
            h_in = hidden_nodes_feat

            position_embedding = self.position_list[i](data.edge_w) if self.position else None
            hidden_nodes_feat = gcn(h_in, data.edge_index, edge_weight=position_embedding)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        return hidden_nodes_feat

    pass


class AttentionClass(nn.Module):

    def __init__(self, in_dim=128, n_classes=10):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1, bias=False)
        self.readout_mlp = nn.Linear(in_dim, n_classes, bias=False)
        pass

    def forward(self, data):
        x = data.x
        x_att = torch.sigmoid(self.attention(x))
        _x = (x_att * x + x) / 2
        hg = global_pool_2(_x, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class PositionEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim, has_act=False):
        super().__init__()
        self.position_embedding_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.position_embedding_2 = nn.Linear(out_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.has_act = has_act
        pass

    def forward(self, x):
        out = self.position_embedding_1(x)
        out = self.relu(out)
        out = self.position_embedding_2(out)
        if self.has_act:
            out = self.sigmoid(out)
        return out

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=6, which=0, has_bn=False,
                 normalize=False, residual=False, improved=False, concat=False, position=True):
        super().__init__()
        self.model_conv = CONVNet(layer_num=conv_layer_num)  # 6, 13
        self.attention_class = AttentionClass(in_dim=128, n_classes=10)

        if which == 1:
            self.model_gnn1 = SAGENet1(in_dim=self.model_conv.features[-2].num_features, hidden_dims=[128, 128],
                                       has_bn=has_bn, normalize=normalize,
                                       residual=residual, concat=concat, position=position)
            self.model_gnn2 = SAGENet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[128, 128, 128, 128],
                                       has_bn=has_bn, normalize=normalize,
                                       residual=residual, concat=concat, position=position)
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
        gcn2_feature = self.model_gnn2.forward(batched_graph)

        batched_graph.x = gcn2_feature
        logits = self.attention_class.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', down_ratio=1, concat=False, which=0,
                 batch_size=64, image_size=32, sp_size=4, train_print_freq=100, test_print_freq=50, lr=None,
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1", conv_layer_num=6, padding=4,
                 has_bn=True, normalize=True, residual=False,
                 improved=False, weight_decay=0.0, is_sgd=False, position=True):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                       is_train=True, image_size=image_size, sp_size=sp_size, padding=padding)
        self.test_dataset = MyDataset(data_root_path=data_root_path, down_ratio=down_ratio,
                                      is_train=False, image_size=image_size, sp_size=sp_size, padding=padding)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=conv_layer_num, which=which, has_bn=has_bn, normalize=normalize,
                              residual=residual, improved=improved, concat=concat, position=position).to(self.device)

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
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  594112  lr:0.01 padding:2 bs:64 Epoch:104, Train:0.9946-1.0000/0.0226 Test:0.9197-0.9961/0.3002
2_2_13  Att *+ ReLU  SGD 2Linear mean_max_pool  494272  lr:0.01 padding:2 bs:64 NoPos 142, Train:0.9950-1.0000/0.0204 Test:0.9188-0.9964/0.3256

NoPos 2020-09-29 19:19:20 Epoch:107, Train:0.9924-1.0000/0.0270 Test:0.9167-0.9976/0.3143
Pos *+ Sigmoid 692416 2020-09-29 17:25:14 Epoch:65, Train:0.9753-0.9999/0.0725 Test:0.9131-0.9964/0.2932

Pos Cat NoSigmoid 790720 Epoch:104, Train:0.9917-1.0000/0.0299 Test:0.9178-0.9969/0.3222
Pos *+ HasSigmoid 692416 Epoch:124, Train:0.9942-1.0000/0.0240 Test:0.9191-0.9968/0.3188
Pos  +  NoSigmoid 692416 
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

    _padding = 2

    _gpu_id = "0"
    # _gpu_id = "1"

    _is_sgd = True

    _position = True
    # _position = False

    # _sp_size, _down_ratio, _conv_layer_num = 4, 1, 6
    _sp_size, _down_ratio, _conv_layer_num = 2, 2, 13

    # global_pool_1, global_pool_2, pool_name = global_mean_pool, global_mean_pool, "mean_mean_pool"
    # global_pool_1, global_pool_2, pool_name = global_max_pool, global_max_pool, "max_max_pool"
    global_pool_1, global_pool_2, pool_name = global_mean_pool, global_max_pool, "mean_max_pool"

    _epochs, _weight_decay, _lr = 150, 5e-4, [[0, 0.1], [50, 0.01], [100, 0.001], [130, 0.0001]]
    _lr = [[0, 0.01], [50, 0.001], [100, 0.0001]]

    _concat, _has_bn, _has_residual, _is_normalize = True, True, True, True
    _improved = True  # No use

    _root_ckpt_dir = "./ckpt2/dgl/1_Position_PYG_CONV_Fast_CIFAR10/{}_{}_{}_{}_{}_{}".format(
        _is_sgd, _sp_size,  _down_ratio, _conv_layer_num, pool_name, _padding)
    Tools.print("epochs:{} ckpt:{} batch size:{} image size:{} sp size:{} down_ratio:{} "
                "conv_layer_num:{} workers:{} gpu:{} has_residual:{} is_normalize:{} "
                "has_bn:{} improved:{} concat:{} is_sgd:{} weight_decay:{} pool_name:{}".format(
        _epochs, _root_ckpt_dir, _batch_size, _image_size, _sp_size, _down_ratio, _conv_layer_num, _num_workers,
        _gpu_id, _has_residual, _is_normalize, _has_bn, _improved, _concat, _is_sgd, _weight_decay, pool_name))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, concat=_concat, which=1,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       residual=_has_residual, normalize=_is_normalize, down_ratio=_down_ratio, lr=_lr,
                       has_bn=_has_bn, improved=_improved, weight_decay=_weight_decay, conv_layer_num=_conv_layer_num,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq, position=_position,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id, padding=_padding)
    runner.train(_epochs)

    pass
