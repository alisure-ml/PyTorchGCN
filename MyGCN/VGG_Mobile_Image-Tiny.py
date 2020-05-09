import os
import torch
import shutil
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2 as mobilenet


class TranTiny(object):

    def __init__(self, val_root="/mnt/4T/Data/tiny-imagenet-200/tiny-imagenet-200/val",
                 tiny_val_txt="val_annotations.txt",
                 val_result_root="/mnt/4T/Data/tiny-imagenet-200/tiny-imagenet-200/val_new"):
        self.val_root = val_root
        self.val_result_root = val_result_root
        self.tiny_val_txt = os.path.join(self.val_root, tiny_val_txt)
        self.tiny_val_image_path = os.path.join(self.val_root, "images")

        self.val_data = self.read_txt()
        pass

    def read_txt(self):
        with open(self.tiny_val_txt) as f:
            tine_val = f.readlines()
            return [i.strip().split("\t")[0:2] for i in tine_val]
            pass
        pass

    def new_val(self):
        for index, (image_name, image_class) in enumerate(self.val_data):
            if index % 100 == 0:
                Tools.print("{} {}".format(index, len(self.val_data)))
            src = os.path.join(self.tiny_val_image_path, image_name)
            dst = Tools.new_dir(os.path.join(self.val_result_root, image_class, image_name))
            shutil.copy(src, dst)
            pass
        pass

    @staticmethod
    def main():
        TranTiny().new_val()
        pass

    pass


class Runner(object):

    def __init__(self, root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=128, image_size=64, lr=0.1):
        self.root_path = root_path
        self.batch_size = batch_size
        self.lr = lr

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = mobilenet(num_classes=200).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0)

        self.train_loader, self.test_loader = self._data(image_size=image_size)
        pass

    def _change_lr(self, epoch):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < 100:
            __change_lr(self.lr)
        elif 100 <= epoch < 150:
            __change_lr(self.lr / 10)
        elif 150 <= epoch:
            __change_lr(self.lr / 100)

        # if 0 <= epoch < 33:
        #     __change_lr(self.lr)
        # elif 33 <= epoch < 66:
        #     __change_lr(self.lr / 10)
        # elif 66 <= epoch:
        #     __change_lr(self.lr / 100)

        pass

    def _data(self, image_size=64):
        Tools.print('==> Preparing data..')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([transforms.Resize(image_size),
                                              transforms.RandomCrop(image_size, padding=8),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])

        _train_dir = os.path.join(self.root_path, "train")
        _test_dir = os.path.join(self.root_path, "val_new")

        train_set = torchvision.datasets.ImageFolder(_train_dir, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)

        test_set = torchvision.datasets.ImageFolder(_test_dir, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)

        return _train_loader, _test_loader

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
     64 SGD 0.1 100 37.007 37.60
    128 SGD 0.1 100 48.084 45.53
    
     64 SGD 0.01 200 69.028 55.15
    128 SGD 0.01 200 81.365 61.24
    """

    # TranTiny(val_root='D:\\data\\ImageNet\\tiny-imagenet-200\\val', tiny_val_txt="val_annotations.txt",
    #          val_result_root="D:\\data\\ImageNet\\tiny-imagenet-200\\val_new").new_val()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1

    _data_root_path = '/mnt/4T/Data/tiny-imagenet-200/tiny-imagenet-200'
    # _data_root_path = '/home/ubuntu/ALISURE/data/tiny-imagenet-200'
    # _data_root_path = 'D:\\data\\ImageNet\\tiny-imagenet-200'

    runner = Runner(root_path=_data_root_path, batch_size=64, image_size=128, lr=0.01)
    runner.info()

    for _epoch in range(runner.start_epoch, 200):
        runner.train(_epoch, change_lr=True)
        runner.test(_epoch)
        pass

    pass
