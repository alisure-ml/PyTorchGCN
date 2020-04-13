import os
import cv2
import dgl
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
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
from visual_embedding_2_norm import DealSuperPixel


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

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, image_size=32):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path
        self.transform = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
                                             transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.data_set = datasets.CIFAR10(root=self.data_root_path, train=self.is_train, transform=self.transform)

        # 3. Super Pixel
        self.image_size = image_size
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        img = np.asarray(img)
        return img, target

    @staticmethod
    def collate_fn(samples):
        imgs, labels = map(list, zip(*samples))
        imgs = torch.tensor(np.transpose(imgs, axes=(0, 3, 1, 2)))
        labels = torch.tensor(np.array(labels))
        return imgs, labels

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


class CNNNet(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = False
        conv_stride = 4
        avg_range = 3

        self.conv0 = ConvBlock(3, 64, stride=conv_stride, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(64, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        # self.conv5 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool5 = nn.AvgPool2d(3, 1, padding=1)
        # self.conv6 = ConvBlock(128, 128, padding=0, ks=1, has_bn=self.has_bn)
        # self.pool6 = nn.AvgPool2d(3, 1, padding=1)
        # self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(128, 10)
        pass

    def forward(self, x):
        e = self.conv0(x)

        e = self.conv1(e)
        e = self.pool1(e)
        e = self.conv2(e)
        e = self.pool2(e)
        e = self.pool_m1(e)

        e = self.conv3(e)
        e = self.pool3(e)
        e = self.conv4(e)
        e = self.pool4(e)
        e = self.pool_m2(e)

        # e = self.conv5(e)
        # e = self.pool5(e)
        # e = self.conv6(e)
        # e = self.pool6(e)
        # e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class CNNNet2(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.has_bn = False
        conv_stride = 4
        avg_range = 3

        self.conv0 = ConvBlock(3, 64, stride=conv_stride, ks=conv_stride, has_bn=self.has_bn)

        self.conv1 = ConvBlock(64, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool1 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv2 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool2 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool3 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.conv4 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool4 = nn.AvgPool2d(avg_range, 1, padding=avg_range//2)
        self.pool_m2 = nn.MaxPool2d(2, 2, padding=0)

        self.conv5 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool5 = nn.AvgPool2d(3, 1, padding=1)
        self.conv6 = ConvBlock(128, 128, padding=1, ks=3, has_bn=self.has_bn)
        self.pool6 = nn.AvgPool2d(3, 1, padding=1)
        self.pool_m3 = nn.MaxPool2d(2, 2, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.readout_mlp = MLPReadout(128, 10)
        pass

    def forward(self, x):
        e = self.conv0(x)

        e = self.conv1(e)
        # e = self.pool1(e)
        e = self.conv2(e)
        # e = self.pool2(e)
        e = self.pool_m1(e)

        e = self.conv3(e)
        # e = self.pool3(e)
        e = self.conv4(e)
        # e = self.pool4(e)
        e = self.pool_m2(e)

        e = self.conv5(e)
        # e = self.pool5(e)
        e = self.conv6(e)
        # e = self.pool6(e)
        e = self.pool_m3(e)

        e = self.avg(e).squeeze()
        out = self.readout_mlp(e)
        return out

    pass


class RunnerSPE(object):

    def __init__(self, model, data_root_path='/mnt/4T/Data/cifar/cifar-10',
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        _image_size = 32

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True, image_size=_image_size)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False, image_size=_image_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)

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

    def _train_epoch(self, print_freq=100):
        self.model.train()
        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for i, (batch_imgs, batch_labels) in enumerate(self.train_loader):
            batch_images = batch_imgs.float().to(self.device)
            batch_labels = batch_labels.long().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model.forward(batch_images)
            loss = self._loss_total(logits, batch_labels)
            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self._accuracy(logits, batch_labels)

            if i % print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_train_acc/nb_data))
                pass
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self, print_freq=50):
        self.model.eval()

        Tools.print()
        epoch_test_acc, nb_data, epoch_test_loss = 0, 0, 0
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.test_loader):
                batch_images = batch_imgs.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                logits = self.model.forward(batch_images)
                loss = self._loss_total(logits, batch_labels)

                nb_data += batch_labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self._accuracy(logits, batch_labels)

                if i % print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1), loss.detach().item(), epoch_test_acc/nb_data))
                    pass
                pass
            pass

        epoch_test_loss /= (len(self.test_loader) + 1)
        epoch_test_acc /= nb_data
        return epoch_test_loss, epoch_test_acc

    def _loss_total(self, logits, batch_labels):
        loss_class = self.loss_class(logits, batch_labels)
        return loss_class

    def _lr(self, epoch):
        if epoch == 25:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
            pass

        if epoch == 50:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0005
            pass

        if epoch == 75:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
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
    # 强数据增强+LR。不确定以下两个哪个带Sigmoid
    GCN  No Sigmoid 2020-04-07 02:50:57 Epoch: 75, lr=0.0000, Train: 0.5148/1.4100 Test: 0.5559/1.3145
    GCN Has Sigmoid 2020-04-07 07:35:40 Epoch: 72, lr=0.0000, Train: 0.5354/1.3428 Test: 0.5759/1.2394
    GCN  No Sigmoid 2020-04-08 06:36:51 Epoch: 70, lr=0.0000, Train: 0.5099/1.4281 Test: 0.5505/1.3224
    GCN Has Sigmoid 2020-04-08 07:24:54 Epoch: 73, lr=0.0001, Train: 0.5471/1.3164 Test: 0.5874/1.2138

    # 原始:数据增强+LR
    GCN           No Sigmoid 2020-04-08 06:24:55 Epoch: 98, lr=0.0001, Train: 0.6696/0.9954 Test: 0.6563/1.0695
    GCN          Has Sigmoid 2020-04-08 15:41:33 Epoch: 97, lr=0.0001, Train: 0.7781/0.6535 Test: 0.7399/0.8137
    GraphSageNet Has Sigmoid 2020-04-08 23:31:25 Epoch: 88, lr=0.0001, Train: 0.8074/0.5703 Test: 0.7612/0.7322
    GatedGCNNet  Has Sigmoid 2020-04-10 03:55:12 Epoch: 92, lr=0.0001, Train: 0.8401/0.4779 Test: 0.7889/0.6741

    CNNNet 2x2 3x3 4layer       2020-04-13 15:18:12 Epoch: 96, lr=0.0001, Train: 0.8049/0.5423 Test: 0.7505/0.7542
    CNNNet 2x2 3x3 4layer       2020-04-13 15:18:12 Epoch: 96, lr=0.0001, Train: 0.8049/0.5423 Test: 0.7505/0.7542
    CNNNet 4x4 3x3 4layer       2020-04-13 13:35:02 Epoch: 99, lr=0.0001, Train: 0.7597/0.6751 Test: 0.7239/0.8237
    CNNNet 6x6 3x3 4layer       2020-04-13 16:18:59 Epoch: 98, lr=0.0001, Train: 0.6807/0.8896 Test: 0.6581/0.9817
    
    CNNNet 4x4 3x3 5layer       2020-04-13 14:28:55 Epoch: 98, lr=0.0001, Train: 0.7594/0.6753 Test: 0.7224/0.8188
    CNNNet 4x4 3x3 4layer       2020-04-13 13:35:02 Epoch: 99, lr=0.0001, Train: 0.7597/0.6751 Test: 0.7239/0.8237
    CNNNet 4x4 3x3 3layer       2020-04-13 14:33:04 Epoch: 95, lr=0.0001, Train: 0.7515/0.6934 Test: 0.7223/0.8088
    
    CNNNet 4x4 3x3 4layer + add 2020-04-13 14:11:01 Epoch: 99, lr=0.0001, Train: 0.7632/0.6645 Test: 0.7285/0.8094
    
    CNNNet 2x2 3x3 5layer       2020-04-13 15:12:25 Epoch: 85, lr=0.0001, Train: 0.8131/0.5320 Test: 0.7515/0.7446
    CNNNet 1x1 3x3 6layer       2020-04-13 16:51:52 Epoch: 97, lr=0.0001, Train: 0.6310/1.0423 Test: 0.5493/1.3548
    CNNNet 2x2 3x3 6layer       2020-04-13 15:29:08 Epoch: 84, lr=0.0001, Train: 0.8114/0.5296 Test: 0.7536/0.7498
    
    CNNNet 4x4 5x5 4layer       2020-04-13 16:22:10 Epoch: 98, lr=0.0001, Train: 0.7357/0.7381 Test: 0.6981/0.8875
    
    CNNNet 2x2 3x3 4layer 2pool 2020-04-13 17:09:29 Epoch: 90, lr=0.0001, Train: 0.8121/0.5278 Test: 0.7654/0.7248
    CNNNet 4x4 3x3 4layer 2pool 2020-04-13 16:49:49 Epoch: 97, lr=0.0001, Train: 0.7588/0.6757 Test: 0.7210/0.8182
    CNNNet 1x1 3x3 4layer 2pool 2020-04-13 19:08:47 Epoch: 77, lr=0.0001, Train: 0.6196/1.0698 Test: 0.5481/1.3280
    CNNNet 2x2 3x3 4layer 2pool 2020-04-13 18:47:49 Epoch: 97, lr=0.0001, Train: 0.8093/0.5348 Test: 0.7624/0.7335
    CNNNet 4x4 3x3 4layer 2pool 2020-04-13 18:39:08 Epoch: 85, lr=0.0001, Train: 0.7521/0.6975 Test: 0.7242/0.8246
    CNNNet 1x1 3x3 6layer 3pool 2020-04-13 19:27:36 Epoch: 97, lr=0.0001, Train: 0.6668/0.9343 Test: 0.5684/1.3213
    CNNNet 2x2 3x3 6layer 3pool 2020-04-13 18:40:26 Epoch: 96, lr=0.0001, Train: 0.8274/0.4786 Test: 0.7625/0.7672
    CNNNet 4x4 3x3 6layer 3pool 2020-04-13 18:36:38 Epoch: 96, lr=0.0001, Train: 0.7562/0.6835 Test: 0.7123/0.8373
    
    CNNNet 1x1 3x3 4layer  3x3  2020-04-13 21:12:20 Epoch: 98, lr=0.0001, Train: 0.9783/0.0610 Test: 0.8743/0.6718
    CNNNet 1x1 3x3 6layer  3x3  2020-04-13 21:11:06 Epoch: 99, lr=0.0001, Train: 0.9713/0.0822 Test: 0.8507/0.7306
    CNNNet 2x2 3x3 4layer  3x3  2020-04-13 19:44:08 Epoch: 96, lr=0.0001, Train: 0.9515/0.1366 Test: 0.8501/0.6302
    CNNNet 2x2 3x3 6layer  3x3  2020-04-13 19:40:44 Epoch: 82, lr=0.0001, Train: 0.9227/0.2124 Test: 0.8353/0.6319
    CNNNet 4x4 3x3 4layer  3x3  2020-04-13 20:44:20 Epoch: 94, lr=0.0001, Train: 0.8873/0.3205 Test: 0.8177/0.6267
    CNNNet 4x4 3x3 6layer  3x3  2020-04-13 21:28:28 Epoch: 97, lr=0.0001, Train: 0.8710/0.3698 Test: 0.8059/0.6274
    
    """
    # _data_root_path = 'D:\data\CIFAR'
    # _root_ckpt_dir = "ckpt2\\dgl\\my\\{}".format("CNNNet")
    # _num_workers = 2
    # _use_gpu = False
    # _gpu_id = "1"

    _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    _root_ckpt_dir = "./ckpt2/dgl/Test_1/{}".format("CNNNet")
    _num_workers = 8
    _use_gpu = True
    _gpu_id = "0"

    Tools.print("ckpt:{}, workers:{}, gpu:{}".format(_root_ckpt_dir, _num_workers, _gpu_id))

    runner = RunnerSPE(model=CNNNet2, data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(100)

    pass
