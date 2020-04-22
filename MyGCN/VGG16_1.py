import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


class CNNNet(nn.Module):

    def __init__(self):
        super().__init__()
        # size = [3, 64, 128, 256, 512]
        size = [3, 64, 128, 128, 128]
        kernel_size = 1
        padding = 0
        self.layers = []
        self.layers.extend([nn.Conv2d(size[0], size[1], kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(size[1]), nn.ReLU(inplace=True)])
        self.layers.extend([nn.Conv2d(size[1], size[1], kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(size[1]), nn.ReLU(inplace=True)])
        self.layers.extend([nn.Conv2d(size[1], size[1], kernel_size=4, padding=0, stride=4),
                            nn.BatchNorm2d(size[1]), nn.ReLU(inplace=True)])

        self.layers.extend([nn.Conv2d(size[1], size[2], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[2]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.Conv2d(size[2], size[2], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[2]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.MaxPool2d(2, 2, padding=0)])

        self.layers.extend([nn.Conv2d(size[2], size[3], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[3]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.Conv2d(size[3], size[3], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[3]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.Conv2d(size[3], size[3], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[3]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.MaxPool2d(2, 2, padding=0)])

        self.layers.extend([nn.Conv2d(size[3], size[4], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[4]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.Conv2d(size[4], size[4], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[4]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.Conv2d(size[4], size[4], kernel_size=kernel_size, padding=padding, stride=1),
                            nn.BatchNorm2d(size[4]), nn.ReLU(inplace=True), nn.AvgPool2d(3, 1, padding=1)])
        self.layers.extend([nn.MaxPool2d(2, 2, padding=0)])

        self.layers.extend([nn.AdaptiveAvgPool2d(1)])

        self.features = nn.Sequential(*self.layers)
        self.readout_mlp = nn.Linear(size[4], 10)
        pass

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.readout_mlp(out)
        return out

    pass


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)
        pass

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg_vgg):
        layers = []
        in_channels = 3
        for x in cfg_vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    pass


class Runner(object):

    def __init__(self, root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=128, lr=0.1):
        self.root_path = root_path
        self.batch_size = batch_size
        self.lr = lr

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = VGG().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_loader, self.test_loader = self._data()
        pass

    def _data(self):
        Tools.print('==> Preparing data..')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.CIFAR10(self.root_path, train=True, download=True, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=True, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        return _train_loader, _test_loader

    def _change_lr(self, epoch):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < 100:
            __change_lr(self.lr)
        elif 100 <= epoch < 200:
            __change_lr(self.lr / 10)
        elif 200 <= epoch:
            __change_lr(self.lr / 100)

        pass

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    def info(self):
        Tools.print("batch size={} lr={}".format(self.batch_size, self.lr))
        Tools.print("{} ".format(self._view_model_param(self.net)))
        pass

    def train(self, epoch, change_lr=False):
        print()
        Tools.print('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch)

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss / len(self.train_loader), 100. * correct / total, correct, total))
        pass

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass
            pass

        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (test_loss / len(self.test_loader), 100. * correct / total, correct, total))
        Tools.print("best_acc={} acc={}".format(self.best_acc, 100. * correct / total))
        pass

    pass


if __name__ == '__main__':
    """
    Ha Pool 86.70
    No Pool 86.31
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1

    runner = Runner(batch_size=128, lr=0.01)
    runner.info()

    # for _epoch in range(runner.start_epoch, 300):
    #     runner.train(_epoch, change_lr=True)
    #     runner.test(_epoch)
    #     pass
    pass
