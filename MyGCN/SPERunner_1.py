import os
import torch
import torch.nn as nn
from SPEGCN_1 import GCN
from SPEData_1 import MyDataset
from SPEUtil_1 import gpu_setup
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from visual_embedding_norm import EmbeddingNetCIFARSmallNorm3


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

        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(data), data.y)
            loss.backward()
            self.optimizer.step()

            Tools.print("{} {} loss={}".format(i, len(self.train_loader), loss.item()))
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
