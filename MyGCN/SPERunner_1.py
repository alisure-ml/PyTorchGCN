import os
import cv2
import torch
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
from torch_geometric.nn import GCNConv, global_mean_pool
from visual_embedding_2_norm import DealSuperPixel, MyCIFAR10, EmbeddingNetCIFARSmallNorm3


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        Tools.print()
        Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda")
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
        g_data = self.get_sp_info(img, target)
        return g_data

    def get_sp_info(self, img, target):
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

        # Data for Batch: super_pixel_info
        x, pos, area, size = [], [], [], []
        for sp_i in range(len(super_pixel_info)):
            now_sp = super_pixel_info[sp_i]

            _size = now_sp["size"]
            _area = now_sp["area"]
            _x = np.concatenate([now_sp["feature_shape"], now_sp["feature_texture"], _size], axis=0)

            x.append(_x)
            size.append([_size])
            area.append(_area)
            pos.append([_area[1] - _area[0], _area[3] - _area[2]])
            pass

        # Data for Batch: adjacency_info
        edge_index, edge_w = [], []
        for edge_i in range(len(adjacency_info)):
            edge_index.append([adjacency_info[edge_i][0], adjacency_info[edge_i][1]])
            edge_w.append([adjacency_info[edge_i][2], adjacency_info[edge_i][2]])
            pass
        edge_index = np.transpose(edge_index, axes=(1, 0))

        # Data for Batch: Data
        g_data = Data(x=torch.from_numpy(np.asarray(x)), edge_index=torch.from_numpy(edge_index),
                      y=torch.tensor([target]), pos=torch.from_numpy(np.asarray(pos)),
                      area=torch.from_numpy(np.asarray(area)), size=torch.from_numpy(np.asarray(size)),
                      edge_w=torch.from_numpy(np.asarray(edge_w)))
        return g_data

    @staticmethod
    def collate_fn(batch_data):
        return Batch.from_data_list(batch_data)

    pass


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.in_dim = 33
        self.hidden_dim = 128
        self.out_dim = 128
        self.n_classes = 10

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)

        self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim, normalize=True)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim, normalize=True)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim, normalize=True)
        self.conv4 = GCNConv(self.hidden_dim, self.out_dim, normalize=True)

        self.read_out1 = nn.Linear(self.out_dim, self.out_dim // 2)
        self.read_out2 = nn.Linear(self.out_dim // 2, self.out_dim // 4)
        self.read_out3 = nn.Linear(self.out_dim // 4, self.n_classes, bias=False)

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        embedding_x = self.embedding_h(data.x)

        gcn_conv1 = self.relu(self.conv1(embedding_x, data.edge_index, data.edge_w))
        gcn_conv2 = self.relu(self.conv2(gcn_conv1, data.edge_index, data.edge_w))
        gcn_conv3 = self.relu(self.conv3(gcn_conv2, data.edge_index, data.edge_w))
        gcn_conv4 = self.relu(self.conv4(gcn_conv3, data.edge_index, data.edge_w))

        aggr_out = global_mean_pool(gcn_conv4, data.batch)

        read_out1 = self.relu(self.read_out1(aggr_out))
        read_out2 = self.relu(self.read_out2(read_out1))
        read_out3 = self.read_out3(read_out2)
        return read_out3

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

        self.model = GCN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss().to(self.device)
        pass

    def train(self, epochs):
        for epoch in range(0, epochs):
            self.train_epoch(epoch)
            test_acc = self.test()
            Tools.print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
            pass
        pass

    def train_epoch(self, epoch):
        self.model.train()

        if epoch == 3:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
            pass

        if epoch == 6:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
            pass

        avg_loss = 0
        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            read_out3 = self.model(data)
            loss = self.loss(read_out3, data.y)
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

            Tools.print("{} {} loss={:4f}/{:4f}".format(i, len(self.train_loader), avg_loss/(i+1), loss.item()))
            pass

        pass

    def test(self):
        self.model.eval()
        correct = 0

        for data in self.test_loader:
            data = data.to(self.device)
            pred = self.model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()
            pass

        return correct / len(self.test_dataset)

    pass


if __name__ == '__main__':

    runner = RunnerSPE()
    runner.train(10)

    pass
