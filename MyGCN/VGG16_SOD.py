import os
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader


class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img, 'label': mask}

    pass


class RandomScaleCrop(object):

    def __init__(self, base_size, crop_size, mask_fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.mask_fill = mask_fill
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # 随机缩放
        ratio = 7 / 8
        short_size = random.randint(int(self.base_size * ratio), int(self.base_size / ratio))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Pad
        if short_size < self.crop_size:
            padh = (self.crop_size - oh if oh < self.crop_size else 0) // 2
            padw = (self.crop_size - ow if ow < self.crop_size else 0) // 2
            img = ImageOps.expand(img, border=(padw, padh, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(padw, padh, padw, padh), fill=self.mask_fill)
            pass

        # Random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size) if w - self.crop_size > 0 else 0
        y1 = random.randint(0, h - self.crop_size) if h - self.crop_size > 0 else 0
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}

    pass


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': img, 'label': mask}
    pass


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path, is_train=True, image_size=320, pool_ratio=8):
        super().__init__()
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // pool_ratio

        # 路径
        self.data_image_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Image" if self.is_train else "DUTS-TE-Image")
        self.data_label_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Mask" if self.is_train else "DUTS-TE-Mask")

        # 数据增强
        self.transform_train = transforms.Compose([FixedResize(self.image_size), RandomHorizontalFlip()])
        self.transform_test = transforms.Compose([FixedResize(self.image_size)])

        # 准备数据
        self.image_name_list, self.label_name_list = self.get_image_label_name()
        pass

    def get_image_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_image_path, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_label_path, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        # return tra_img_name_list[:12], tra_lbl_name_list[:12]
        return tra_img_name_list, tra_lbl_name_list

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        label = Image.open(self.label_name_list[idx])
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        # 数据增强
        sample = {'image': image, 'label': label}
        sample = self.transform_train(sample) if self.is_train else self.transform_test(sample)
        image, label = sample['image'], sample['label']

        # 归一化
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image)

        # 标签
        label_small_data = np.asarray(label.resize((self.image_size_for_sp, self.image_size_for_sp))) / 255
        label = np.asarray(label) / 255

        # 返回
        return img_data, label, label_small_data

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


class GCNBlock(nn.Module):

    def __init__(self, cin=146, cout=146, padding=0, ks=1, has_relu=True, has_bn=True):
        super().__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.pool = nn.AvgPool2d(3, 1, padding=1)
        self.conv = ConvBlock(cin, cout, padding=padding, ks=ks, has_bn=self.has_bn, has_relu=self.has_relu)
        pass

    def forward(self, e):
        e = self.conv(e)
        e = self.pool(e)
        return e

    pass


class VGG2(nn.Module):

    def __init__(self):
        super(VGG2, self).__init__()
        self.features = self._make_layers()
        self.classifier = ConvBlock(256, 1, ks=1, padding=0, has_relu=False, has_bn=False)
        pass

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        out = out.squeeze(1)
        return out, torch.sigmoid(out)

    @staticmethod
    def _make_layers():
        layers = [
            ConvBlock(3, 64, ks=3, padding=1),  # 320
            ConvBlock(64, 64, ks=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 160
            ConvBlock(64, 128, ks=3, padding=1),
            ConvBlock(128, 128, ks=3, padding=1),

            GCNBlock(128, 128, ks=1, padding=0),
            GCNBlock(128, 128, ks=1, padding=0),
            nn.AvgPool2d(kernel_size=4, stride=4),  # 40

            GCNBlock(128, 256, ks=1, padding=0),
            GCNBlock(256, 256, ks=1, padding=0),
            GCNBlock(256, 256, ks=1, padding=0),
            GCNBlock(256, 256, ks=1, padding=0),
        ]
        return nn.Sequential(*layers)

    pass


class Runner(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=128, image_size=320,
                 lr=0.1, pool_ratio=8, train_print_freq=100, test_print_freq=50, num_workers=8):
        self.data_root_path = data_root_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = lr
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = VGG2().to(self.device)
        self.criterion = nn.BCELoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0)

        self.train_dataset = MyDataset(data_root_path, is_train=True, image_size=image_size, pool_ratio=pool_ratio)
        self.test_dataset = MyDataset(data_root_path, is_train=False, image_size=image_size, pool_ratio=pool_ratio)

        self.train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size, shuffle=False, num_workers=num_workers)
        pass

    def _change_lr(self, epoch):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < 33:
            __change_lr(self.lr)
        elif 33 <= epoch < 66:
            __change_lr(self.lr / 10)
        elif 66 <= epoch:
            __change_lr(self.lr / 100)

        pass

    def load_model(self, model_file_name):
        self.net.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    @staticmethod
    def save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    def info(self):
        Tools.print("batch size={} lr={}".format(self.batch_size, self.lr))
        Tools.print("{} ".format(self._view_model_param(self.net)))
        pass

    def train(self, epoch, change_lr=False):
        Tools.print()
        Tools.print('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch)

        self.net.train()
        th_num = 100
        epoch_loss, nb_data = 0, 0
        epoch_mae = 0.0
        epoch_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall = np.zeros(shape=(th_num,)) + 1e-6
        for batch_idx, (inputs, targets, targets_small) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets_small = targets_small.float().to(self.device)
            self.optimizer.zero_grad()
            logits, logits_sigmoid = self.net(inputs)
            loss = self.criterion(logits_sigmoid, targets_small)
            labels_val = targets_small.cpu().detach().numpy()
            logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()
            loss.backward()
            self.optimizer.step()

            nb_data += targets_small.size(0)
            epoch_loss += loss.detach().item()
            mae = self._eval_mae(logits_sigmoid_val, labels_val)
            prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # Print
            if batch_idx % self.train_print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}/{:4f}".format(
                    batch_idx, len(self.train_loader), epoch_loss/(batch_idx+1),
                    loss.detach().item(), epoch_mae/(batch_idx+1), mae))
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_loss / len(self.train_loader), epoch_mae / len(self.train_loader)
        _avg_prec, _avg_recall = epoch_prec / len(self.train_loader), epoch_recall / len(self.train_loader)
        score = (1 + 0.3) * _avg_prec * _avg_recall / (0.3 * _avg_prec + _avg_recall)
        return avg_loss, avg_mae, score.max()

    def test(self):
        self.net.eval()
        th_num = 100
        epoch_test_loss, nb_data = 0, 0
        epoch_test_mae = 0.0
        epoch_test_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6
        with torch.no_grad():
            for batch_idx, (inputs, targets, targets_small) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_small = targets_small.float().to(self.device)
                logits, logits_sigmoid = self.net(inputs)
                loss = self.criterion(logits_sigmoid, targets_small)
                labels_val = targets_small.cpu().detach().numpy()
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += targets_small.size(0)
                epoch_test_loss += loss.detach().item()

                mae = self._eval_mae(logits_sigmoid_val, labels_val)
                prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)

                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # Print
                if batch_idx % self.test_print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}/{:4f}".format(
                        batch_idx, len(self.test_loader), epoch_test_loss/(batch_idx+1),
                        loss.detach().item(), epoch_test_mae/(batch_idx+1), mae))
                    pass
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_test_loss / len(self.test_loader), epoch_test_mae / len(self.test_loader)
        _avg_prec, _avg_recall = epoch_test_prec / len(self.test_loader), epoch_test_recall / len(self.test_loader)
        score = (1 + 0.3) * _avg_prec * _avg_recall / (0.3 * _avg_prec + _avg_recall)
        return avg_loss, avg_mae, score.max()

    def visual(self, model_file=None):
        if model_file:
            self.load_model(model_file_name=model_file)
            pass

        self.net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, targets_small) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_small = targets_small.float().to(self.device)
                logits, logits_sigmoid = self.net(inputs)
                labels_val = targets_small.cpu().detach().numpy()
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()

                for i in range(targets_small.size(0)):
                    pre = np.asarray(logits_sigmoid_val[i] * 255, dtype=np.uint8)
                    Image.fromarray(pre).show()

                    lab = np.asarray(labels_val[i] * 255, dtype=np.uint8)
                    Image.fromarray(lab).show()
                    pass
                pass
            pass

        pass

    @staticmethod
    def _eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def _eval_pr(y_pred, y, th_num=100):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    pass


if __name__ == '__main__':
    """
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1

    # _data_root_path = 'D:\\data\\SOD\\DUTS'
    _data_root_path = '/home/ubuntu/ALISURE/data/SOD/DUTS'
    _root_ckpt_dir = Tools.new_dir("./ckpt3/dgl/6_DGL_VGG16_SOD")
    runner = Runner(data_root_path=_data_root_path, batch_size=16, lr=0.01,
                    num_workers=4, train_print_freq=100, test_print_freq=50)
    runner.info()

    # runner.visual(model_file="./ckpt3/dgl/6_DGL_VGG16_SOD/epoch_1.pkl")
    runner.visual()

    for _epoch in range(runner.start_epoch, 100):
        epoch_loss, epoch_train_mae, epoch_train_score = runner.train(_epoch, change_lr=True)
        runner.save_checkpoint(runner.net, _root_ckpt_dir, _epoch)
        epoch_test_loss, epoch_test_mae, epoch_test_score = runner.test()
        Tools.print('E:{:02d}, Train(mae-score-loss):{:.4f}/{:.4f}/{:.4f} Test(mae-score-loss):'
                    '{:.4f}/{:.4f}/{:.4f}'.format(_epoch, epoch_train_mae, epoch_train_score, epoch_loss,
                                                  epoch_test_mae, epoch_test_score, epoch_test_loss))
        pass

    pass
