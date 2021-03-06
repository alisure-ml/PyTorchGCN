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
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from layers.mlp_readout_layer import MLPReadout
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling, EdgePooling, SAGPooling


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
                                         sigma=slic_sigma, max_iter=slic_max_iter)
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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // 1
        self.data_root_path = data_root_path

        self.transform = transforms.Compose([transforms.RandomCrop(self.image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        img_data = transforms.Compose([transforms.ToTensor()])(np.asarray(img)).unsqueeze(dim=0)

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
                     edge_w=torch.from_numpy(sp_adj[1]).unsqueeze(1).float())
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([target]),
                               edge_w=torch.from_numpy(super_pixel[2]).unsqueeze(1).float())
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

    def __init__(self, in_dim=128, hidden_dims=[146, 146, 146, 146], out_dim=146,
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNConv(self.hidden_dims[-1], out_dim, normalize=self.normalize, improved=self.improved))

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims[1:]:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            self.bn_list.append(nn.BatchNorm1d(out_dim))
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_mean_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10,
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNConv(self.hidden_dims[-1], out_dim, normalize=self.normalize, improved=self.improved))

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims[1:]:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            self.bn_list.append(nn.BatchNorm1d(out_dim))
            pass

        self.readout_mlp = MLPReadout(out_dim, n_classes)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass
        hg = global_mean_pool(hidden_nodes_feat, data.batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class GCNNetTopK1(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[146, 146, 146, 146], out_dim=146, normalize=False):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=normalize))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNConv(self.hidden_dims[-1], out_dim, normalize=normalize))
        self.relu = nn.ReLU()

        self.top_k = TopKPooling(out_dim, ratio=0.7)
        pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding_h(x)
        for gcn in self.gcn_list:
            x = self.relu(gcn(x, edge_index))
            pass
        x, edge_index, _, batch, _, _ = self.top_k(x, edge_index, None, batch)
        hg = global_mean_pool(x, batch)
        return hg

    pass


class GCNNetTopK2(nn.Module):

    def __init__(self, in_dim=146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10, normalize=False):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.embedding_h = nn.Linear(in_dim, self.hidden_dims[0])

        self.gcn_list = nn.ModuleList()
        _in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=normalize))
            _in_dim = hidden_dim
            pass
        self.gcn_list.append(GCNConv(self.hidden_dims[-1], out_dim, normalize=normalize))

        self.readout_mlp = MLPReadout(out_dim, n_classes)
        self.relu = nn.ReLU()

        self.top_k = TopKPooling(out_dim, ratio=0.7)
        pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding_h(x)
        for gcn in self.gcn_list:
            x = self.relu(gcn(x, edge_index))
            pass

        x, edge_index, _, batch, _, _ = self.top_k(x, edge_index, None, batch)

        hg = global_mean_pool(x, batch)
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self, has_bn=False, normalize=True, residual=False, improved=False):
        super().__init__()
        self.model_conv = CONVNet(in_dim=3, hidden_dims=[64, 64], out_dim=64)

        self.model_gnn1 = GCNNet1(in_dim=64, hidden_dims=[146, 146], out_dim=146,
                                  has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
        self.model_gnn2 = GCNNet2(146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10,
                                  has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)

        # self.model_gnn1 = GCNNetTopK1(in_dim=64, hidden_dims=[146, 146], out_dim=146)
        # self.model_gnn2 = GCNNetTopK2(in_dim=146, hidden_dims=[146, 146, 146, 146], out_dim=146, n_classes=10)
        pass

    def forward(self, images, batched_graph, batched_pixel_graph):
        # model 1
        conv_feature = self.model_conv(images) if self.model_conv else images

        # model 2
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = conv_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph)

        # model 3
        batched_graph.x = gcn1_feature
        logits = self.model_gnn2.forward(batched_graph)
        return logits

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=64, image_size=32, sp_size=4,
                 train_print_freq=100, test_print_freq=50, root_ckpt_dir="./ckpt2/norm3",
                 num_workers=8, use_gpu=True, gpu_id="1",
                 has_bn=True, normalize=True, residual=False, improved=False, weight_decay=0.0, is_sgd=False):
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

        self.model = MyGCNNet(has_bn=has_bn, normalize=normalize, residual=residual, improved=improved).to(self.device)

        if is_sgd:
            self.lr_s = [[0, 0.1], [50, 0.01], [100, 0.001]]
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr_s[0][1],
                                             momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1],
                                              weight_decay=weight_decay)
            pass
        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))

        self.loss_class = nn.CrossEntropyLoss().to(self.device)
        pass

    def load_model(self, model_file_name):
        self.model.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def _lr(self, epoch):
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
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
        for i, (images, labels, batched_graph, batched_pixel_graph) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels = labels.long().to(self.device)

            batched_graph.batch = batched_graph.batch.to(self.device)
            batched_graph.edge_index = batched_graph.edge_index.to(self.device)

            batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
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
            for i, (images, labels, batched_graph, batched_pixel_graph) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                labels = labels.long().to(self.device)

                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                # Run
                logits = self.model.forward(images, batched_graph, batched_pixel_graph)
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
    GCN        249521 2020-05-02 20:47:54 Epoch: 77, lr=0.0000, Train: 0.9454/0.1538 Test: 0.8608/0.5652
    GCNNetTopK 249813 2020-05-03 08:08:14 Epoch: 76, lr=0.0001, Train: 0.8509/0.4303 Test: 0.8020/0.6302
    
    
    GCN 249521 Adam has_residual:True  is_normalize:True  has_bn:False improved:False is_sgd:False weight_decay=0.0    Epoch: 88, lr=0.0000, Train: 0.9628/0.1011 Test: 0.8640/0.5662
    GCN 249521 Adam has_residual:True  is_normalize:True  has_bn:False improved:False is_sgd:False weight_decay=0.0001 Epoch: 78, lr=0.0000, Train: 0.9379/0.1770 Test: 0.8705/0.4384
    GCN 249521 Adam has_residual:False is_normalize:True  has_bn:False improved:False is_sgd:False weight_decay=0.0    Epoch: 83, lr=0.0000, Train: 0.9112/0.2528 Test: 0.8443/0.5242
    GCN 249521 Adam has_residual:False is_normalize:False has_bn:False improved:False is_sgd:False weight_decay=0.0    Epoch: 81, lr=0.0000, Train: 0.9462/0.1530 Test: 0.8548/0.5986
    
    GCN 251273 Adam has_residual:True  is_normalize:True  has_bn:True  improved:False is_sgd:False weight_decay:0.0    Epoch: 98, lr=0.0000, Train: 0.9636/0.0988 Test: 0.8765/0.4846
    GCN 251273 Adam has_residual:True  is_normalize:False has_bn:True  improved:False is_sgd:False weight_decay:0.0    Epoch: 86, lr=0.0000, Train: 0.9569/0.1204 Test: 0.8731/0.4839
    GCN 251273 Adam has_residual:True  is_normalize:True  has_bn:True  improved:True  is_sgd:False weight_decay:0.0    Epoch: 85, lr=0.0000, Train: 0.9648/0.1021 Test: 0.8810/0.4756
    GCN 251273 Adam has_residual:True  is_normalize:True  has_bn:True  improved:False is_sgd:False weight_decay:0.0005 Epoch: 90, lr=0.0000, Train: 0.9373/0.1821 Test: 0.8688/0.4450
    GCN 251273 Adam has_residual:True  is_normalize:False has_bn:True  improved:False is_sgd:False weight_decay:0.0005 Epoch: 89, lr=0.0000, Train: 0.9319/0.1979 Test: 0.8743/0.4240
    GCN 251273 Adam has_residual:True  is_normalize:True  has_bn:True  improved:True  is_sgd:False weight_decay:0.0005 Epoch: 86, lr=0.0000, Train: 0.9313/0.2004 Test: 0.8647/0.4438
    
    GCN 251273 SGD  has_residual:True  is_normalize:True  has_bn:True  improved:True  is_sgd:True  weight_decay:0.0    Epoch:107, lr=0.0010, Train: 0.9331/0.1865 Test: 0.8666/0.4665
    GCN 251273 SGD  has_residual:True  is_normalize:True  has_bn:True  improved:False is_sgd:True  weight_decay:0.0    Epoch:113, lr=0.0010, Train: 0.9324/0.1898 Test: 0.8656/0.4671
    GCN 251273 SGD  has_residual:True  is_normalize:True  has_bn:True  improved:False is_sgd:True  weight_decay:0.0005 Epoch:113, lr=0.0010, Train: 0.9393/0.1744 Test: 0.8837/0.3818
    GCN 251273 SGD  has_residual:True  is_normalize:True  has_bn:True  improved:True  is_sgd:True  weight_decay:0.0005 Epoch:134, lr=0.0010, Train: 0.9562/0.1295 Test: 0.8845/0.3950
    
    1、Adam 不加 weight_decay，SGD 加 weight_decay
    2、improved 有效，但是不多
    3、残差 效果大
    
    结论：SGD+所有+weight_decay | Adam+所有+不加weight_decay
    """
    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("GCNNet")
    # _batch_size = 64
    # _image_size = 32
    # _sp_size = 4
    # _epochs = 100
    # _train_print_freq = 1
    # _test_print_freq = 1
    # _num_workers = 1
    # _use_gpu = False
    # _gpu_id = "1"

    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    # _data_root_path = '/home/ubuntu/ALISURE/data/cifar'
    _root_ckpt_dir = "./ckpt2/dgl/1_PYG_CONV_Fast/{}".format("GCNNet")
    _batch_size = 64
    _image_size = 32
    _sp_size = 4
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 20
    _use_gpu = True

    _gpu_id = "0"
    # _gpu_id = "1"

    # _epochs = 100
    # _is_sgd = False
    _weight_decay = 0.0

    _epochs = 150
    _is_sgd = True
    # _weight_decay = 5e-4

    _improved = False
    _has_bn = True
    _has_residual = True
    _is_normalize = True

    Tools.print("epochs:{} ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{} "
                "has_residual:{} is_normalize:{} has_bn:{} improved:{} is_sgd:{} weight_decay:{}".format(
        _epochs, _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id,
        _has_residual, _is_normalize, _has_bn, _improved, _is_sgd, _weight_decay))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       residual=_has_residual, normalize=_is_normalize,
                       has_bn=_has_bn, improved=_improved, weight_decay=_weight_decay,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs)

    pass
