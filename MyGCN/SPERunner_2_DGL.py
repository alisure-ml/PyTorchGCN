import os
import cv2
import dgl
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
from layers.gcn_layer import GCNLayer
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from layers.mlp_readout_layer import MLPReadout
from torch.utils.data import Dataset, DataLoader
from visual_embedding_2_norm import DealSuperPixel, MyCIFAR10, EmbeddingNetCIFARSmallNorm3


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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, device=None,
                 ve_model_file_name="ckpt\\norm3\\epoch_1.pkl", VEModel=EmbeddingNetCIFARSmallNorm3,
                 image_size=32, sp_size=4, sp_ve_size=6):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path
        self.ve_transform = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
                                                transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.ve_data_set = MyCIFAR10(root=self.data_root_path, train=self.is_train, transform=self.ve_transform)

        # 2. Model
        self.device = torch.device("cpu") if device is None else device
        self.ve_model_file_name = ve_model_file_name
        self.ve_model = VEModel(is_train=False).to(self.device)
        self.ve_model.load_state_dict(torch.load(self.ve_model_file_name), strict=False)

        # 3. Super Pixel
        self.image_size = image_size
        self.sp_size = sp_size
        self.sp_ve_size = sp_ve_size
        pass

    def __len__(self):
        return len(self.ve_data_set)

    def __getitem__(self, idx):
        img, target = self.ve_data_set.__getitem__(idx)
        graph, target = self.get_sp_info(img, target)
        return graph, target

    def get_sp_info(self, img, target, is_add_self=False):
        # 3. Super Pixel
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size, super_pixel_size=self.sp_size)
        segment, super_pixel_info, adjacency_info = deal_super_pixel.run()

        # Resize Super Pixel
        _now_data_list = []
        for key in super_pixel_info:
            _now_data = cv2.resize(super_pixel_info[key]["data2"] / 255,
                                   (self.sp_ve_size, self.sp_ve_size), interpolation=cv2.INTER_NEAREST)
            _now_data_list.append(_now_data)
            pass
        net_data = np.transpose(_now_data_list, axes=(0, 3, 1, 2))

        # 4. Visual Embedding
        shape_feature, texture_feature = self.ve_model.forward(torch.from_numpy(net_data).float().to(self.device))
        shape_feature, texture_feature = shape_feature.detach().numpy(), texture_feature.detach().numpy()
        for sp_i in range(len(super_pixel_info)):
            super_pixel_info[sp_i]["feature_shape"] = shape_feature[sp_i]
            super_pixel_info[sp_i]["feature_texture"] = texture_feature[sp_i]
            pass

        #Graph
        graph = dgl.DGLGraph()

        # Node
        x, pos, area, size = [], [], [], []
        for sp_i in range(len(super_pixel_info)):
            now_sp = super_pixel_info[sp_i]

            _size = now_sp["size"]
            _area = now_sp["area"]
            _x = np.concatenate([now_sp["feature_shape"], now_sp["feature_texture"], [_size]], axis=0)

            x.append(_x)
            size.append([_size])
            area.append(_area)
            pos.append([_area[1] - _area[0], _area[3] - _area[2]])
            pass
        x = np.asarray(x)
        pos = np.asarray(pos)
        size = np.asarray(size)
        area = np.asarray(area)

        # Node Add
        graph.add_nodes(x.shape[0])
        graph.ndata['feat'] = torch.from_numpy(x).half()

        # Edge
        edge_index, edge_w = [], []
        for edge_i in range(len(adjacency_info)):
            edge_index.append([adjacency_info[edge_i][0], adjacency_info[edge_i][1]])
            edge_w.append(adjacency_info[edge_i][2])
            pass
        edge_index = np.asarray(edge_index)
        edge_w = np.asarray(edge_w)

        # Edge Add
        graph.add_edges(edge_index[:, 0], edge_index[:, 1])
        graph.edata['feat'] = torch.from_numpy(edge_w).unsqueeze(1).half()

        if is_add_self:
            graph = self.self_loop(graph)
            pass

        return graph, target

    @staticmethod
    def collate_fn(samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))

        nodes_num = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        edges_num = [graphs[i].number_of_edges() for i in range(len(graphs))]
        nodes_num_norm = [torch.zeros((num, 1)).fill_(1. / float(num)) for num in nodes_num]
        edges_num_norm = [torch.zeros((num, 1)).fill_(1. / float(num)) for num in edges_num]
        nodes_num_norm_sqrt = torch.cat(nodes_num_norm).sqrt()
        edges_num_norm_sqrt = torch.cat(edges_num_norm).sqrt()

        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
            pass

        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, nodes_num_norm_sqrt, edges_num_norm_sqrt

    @staticmethod
    def self_loop(g):
        new_g = dgl.DGLGraph()
        new_g.add_nodes(g.number_of_nodes())
        new_g.ndata['feat'] = g.ndata['feat']

        src, dst = g.all_edges(order="eid")
        src = dgl.backend.zerocopy_to_numpy(src)
        dst = dgl.backend.zerocopy_to_numpy(dst)
        non_self_edges_idx = src != dst
        nodes = np.arange(g.number_of_nodes())
        new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
        new_g.add_edges(nodes, nodes)
        new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
        return new_g

    pass


class GCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.L = 4
        self.readout = "mean"
        self.in_dim = 33
        self.hidden_dim = 146
        self.out_dim = 146
        self.n_classes = 10
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        self.residual = True

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


class MLPNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.L = 4
        self.in_dim = 33
        self.hidden_dim = 168
        self.n_classes = 10
        self.dropout = 0.0
        self.in_feat_dropout = 0.0
        self.gated = False

        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        feat_mlp_modules = [nn.Linear(self.in_dim, self.hidden_dim, bias=True), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(net_params.L - 1):  # L=4
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


class RunnerSPE(object):

    def __init__(self):
        self.device = gpu_setup(use_gpu=False, gpu_id="0")

        _data_root_path = 'D:\data\CIFAR'
        _ve_model_file_name = "ckpt\\norm3\\epoch_1.pkl"
        _VEModel = EmbeddingNetCIFARSmallNorm3
        _image_size = 32
        _sp_size = 4
        _sp_ve_size = 6
        self.root_ckpt_dir = Tools.new_dir("ckpt2\\norm3")

        self.train_dataset = MyDataset(data_root_path=_data_root_path, is_train=True, device=self.device,
                                       ve_model_file_name=_ve_model_file_name, VEModel=_VEModel,
                                       image_size=_image_size, sp_size=_sp_size, sp_ve_size=_sp_ve_size)
        self.test_dataset = MyDataset(data_root_path=_data_root_path, is_train=False, device=self.device,
                                      ve_model_file_name=_ve_model_file_name, VEModel=_VEModel,
                                      image_size=_image_size, sp_size=_sp_size, sp_ve_size=_sp_ve_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True,
                                       num_workers=2, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False,
                                      num_workers=2, collate_fn=self.test_dataset.collate_fn)

        self.model = GCNNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss().to(self.device)
        pass

    def train(self, epochs):
        for epoch in range(0, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            epoch_loss, epoch_train_acc = self.train_epoch(epoch)
            self.save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc = self.test()

            Tools.print('Epoch: {:02d}, lr={:.4f}, Train: {:.4f}/{:.4f} Test: {:.4f}/{:.4f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'],
                epoch_train_acc, epoch_loss, epoch_test_acc, epoch_test_loss))
            pass
        pass

    def train_epoch(self, epoch):
        if epoch == 3:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
            pass

        if epoch == 6:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
            pass

        self.model.train()
        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for i, (batch_graphs, batch_labels,
                batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(self.train_loader):
            batch_nodes_feat = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_edges_feat = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.long().to(self.device)
            batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.device)  # num x 1
            batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.device)

            self.optimizer.zero_grad()
            batch_scores = self.model.forward(batch_graphs, batch_nodes_feat, batch_edges_feat,
                                              batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
            loss = self.loss(batch_scores, batch_labels)
            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self.accuracy(batch_scores, batch_labels)

            Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_train_acc/nb_data))
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self):
        self.model.eval()

        Tools.print()
        epoch_test_loss, epoch_test_acc, nb_data = 0, 0, 0
        with torch.no_grad():
            for i, (batch_graphs, batch_labels,
                    batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(self.test_loader):
                batch_nodes_feat = batch_graphs.ndata['feat'].to(self.device)
                batch_edges_feat = batch_graphs.edata['feat'].to(self.device)
                batch_labels = batch_labels.long().to(self.device)
                batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.device)
                batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.device)

                batch_scores = self.model.forward(batch_graphs, batch_nodes_feat, batch_edges_feat,
                                                  batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
                loss = self.loss(batch_scores, batch_labels)

                nb_data += batch_labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self.accuracy(batch_scores, batch_labels)

                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.test_loader), epoch_test_loss/(i+1), loss.detach().item(), epoch_test_acc/nb_data))
                pass
            pass

        epoch_test_loss /= (len(self.test_loader) + 1)
        epoch_test_acc /= nb_data
        return epoch_test_loss, epoch_test_acc

    def load_model(self, model_file_name):
        self.model.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    @staticmethod
    def save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    @staticmethod
    def accuracy(scores, targets):
        return (scores.detach().argmax(dim=1) == targets).float().sum().item()

    @staticmethod
    def view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':

    runner = RunnerSPE()
    runner.load_model("ckpt2\\norm3\\epoch_0.pkl")
    _test_loss, _test_acc = runner.test()
    Tools.print('Test: {:.4f}/{:.4f}'.format(_test_acc, _test_loss))
    runner.train(10)

    pass
