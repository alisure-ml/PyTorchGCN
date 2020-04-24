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


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4):
        super().__init__()
        self.is_train = is_train
        self.data_root_path = data_root_path
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // 1
        self.sp_size = sp_size
        assert self.image_size_for_sp % self.sp_size == 0

        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)

        self.graph, self.pixel_graph = self.get_sp_info(self.image_size_for_sp, self.sp_size)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # img_data = transforms.Compose([transforms.ToTensor(), normalize])(np.asarray(img)).unsqueeze(dim=0)
        img_data = transforms.Compose([transforms.ToTensor()])(np.asarray(img)).unsqueeze(dim=0)
        return self.graph, self.pixel_graph, img_data, target

    @classmethod
    def get_sp_info(cls, image_size_for_sp, sp_size):
        # Super Pixel
        #################################################################################
        sp_adj, pixel_adj = cls.get_adj(image_size=image_size_for_sp, sp_size=sp_size)
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

    @classmethod
    def get_adj(cls, image_size, sp_size):
        sp_num = image_size // sp_size

        data_where, edge_index, edge_w = cls._adj(sp_num)
        sp_adj = [np.asarray(edge_index), np.asarray(edge_w)]

        pixel_adj = []
        for i in range(sp_num):
            for j in range(sp_num):
                pixel_data_where, pixel_edge_index, pixel_edge_w = cls._adj(sp_size, sp_size * i, sp_size * j)
                pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w])
                pass

        return sp_adj, pixel_adj

    @staticmethod
    def _adj(num, s_i=0, s_j=0, k=3):
        _now_where = [[_i + s_i, _j + s_j] for _i in range(num) for _j in range(num)]

        _a = np.tile([_now_where], (len(_now_where), 1, 1))
        _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
        _dis[_dis == 0] = 111
        _dis = _dis <= 2*((k//2)**2)
        edge_index = np.argwhere(_dis)

        edge_w = np.ones(len(edge_index))
        data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
        return data_where, edge_index, edge_w

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
        self.dropout = 0.0
        self.residual = True
        self.graph_norm = True
        self.batch_norm = True

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])
        if len(self.hidden_dims) >= 2:
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

        if len(self.hidden_dims) >= 2:
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
        if len(self.hidden_dims) >= 3:
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


class MyGCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64], out_dim=64)
        self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[146, 146], out_dim=146)
        self.model_gnn2 = GCNNet2(in_dim=146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10)
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
    GCN      Baseline Has Sigmoid  2020-04-08 15:41:33 Epoch: 97, Train: 0.7781/0.6535 Test: 0.7399/0.8137
    
    """
    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("GCNNet")
    # _batch_size = 64
    # _image_size = 32
    # _sp_size = 4
    # _epochs = 300
    # _train_print_freq = 1
    # _test_print_freq = 1
    # _num_workers = 1
    # _use_gpu = False
    # _gpu_id = "1"

    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    # _data_root_path = '/home/ubuntu/ALISURE/data/cifar'
    _root_ckpt_dir = "./ckpt2/dgl/4_DGL_CONV/{}-small-sgd-lr-300".format("GCNNet")
    _batch_size = 128
    _image_size = 32
    _sp_size = 4
    _epochs = 300
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 8
    _use_gpu = True
    _gpu_id = "0"
    # _gpu_id = "1"

    Tools.print("ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs)

    pass
