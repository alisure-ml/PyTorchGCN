import os
import glob
import torch
import random
import skimage
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torch_geometric.nn import global_mean_pool, SAGEConv


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


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}

    pass


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        self.size = (int(size), int(size))
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        i, j, h, w = self.get_params(img, self.size)
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return {'image': img, 'label': mask}

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path, down_ratio=4, down_ratio2=1, is_train=True, sp_size=4, min_size=256):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.down_ratio_for_sp = down_ratio
        self.down_ratio_for_sod = down_ratio2
        self.data_root_path = data_root_path
        self.min_size = min_size

        # 路径
        self.data_image_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Image" if self.is_train else "DUTS-TE-Image")
        self.data_label_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Mask" if self.is_train else "DUTS-TE-Mask")

        # 数据增强
        self.transform_train = transforms.Compose([RandomHorizontalFlip()])

        # 准备数据
        self.image_name_list, self.label_name_list = self.get_image_label_name()
        pass

    def get_image_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_image_path, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_label_path, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        return tra_img_name_list, tra_lbl_name_list

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        label = Image.open(self.label_name_list[idx])
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        image_name = self.image_name_list[idx]
        if image.size == label.size:
            # 限制最小大小
            if image.size[0] < self.min_size or image.size[1] < self.min_size:
                if image.size[0] < image.size[1]:
                    image = image.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
                    label = label.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
                else:
                    image = image.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
                    label = label.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
                pass

            w, h = label.size
            # 数据增强
            sample = {'image': image, 'label': label}
            sample = self.transform_train(sample) if self.is_train else sample
            image, label = sample['image'], sample['label']
            label_for_sod = np.asarray(label.resize((w//self.down_ratio_for_sod, h//self.down_ratio_for_sod))) / 255

            # 归一化
            _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

            # 超像素
            image_small_data = np.asarray(image.resize((w//self.down_ratio_for_sp, h//self.down_ratio_for_sp)))
            label_for_sp = np.asarray(label.resize((w//self.down_ratio_for_sp, h//self.down_ratio_for_sp))) / 255
        else:
            Tools.print('IMAGE ERROR, PASSING {}'.format(image_name))
            img_data, label_for_sp, label_for_sod, image_small_data, image_name = self.__getitem__(
                np.random.randint(0, len(self.image_name_list)))
            pass
        # 返回
        return img_data, label_for_sp, label_for_sod, image_small_data, image_name

    @staticmethod
    def collate_fn(samples):
        images, labels_sp, labels_sod, images_small, image_name = map(list, zip(*samples))

        images = torch.cat(images)
        images_small = torch.tensor(images_small)

        return images, labels_sp, labels_sod, images_small, image_name

    pass


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, ks=3, has_relu=True, has_bn=False, bias=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
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


class SAGENet1(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.concat = concat
        self.out_num = self.hidden_dims[-1]

        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass
        pass

    def forward(self, data):
        hidden_nodes_feat = data.x
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_mean_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class SAGENet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], skip_which=[1, 2, 3],
                 skip_dim=128, sout=1, has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat
        self.out_num = skip_dim

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass

        # skip
        self.skip_connect_index = skip_which
        self.skip_connect_list = nn.ModuleList()
        self.skip_connect_bn_list = nn.ModuleList()
        for hidden_dim in [self.hidden_dims[which-1] for which in skip_which]:
            self.skip_connect_list.append(nn.Linear(hidden_dim, skip_dim, bias=False))
            self.skip_connect_bn_list.append(nn.BatchNorm1d(skip_dim))
            pass

        self.readout_mlp = nn.Linear(self.out_num, sout, bias=False)
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        gcn_hidden_nodes_feat = [hidden_nodes_feat]
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat

            gcn_hidden_nodes_feat.append(hidden_nodes_feat)
            pass

        skip_connect = []
        for sc, index, bn in zip(self.skip_connect_list, self.skip_connect_index, self.skip_connect_bn_list):
            # Conv
            sc_feat = sc(gcn_hidden_nodes_feat[index])
            if self.has_bn:
                sc_feat = bn(sc_feat)
            sc_feat = self.relu(sc_feat)

            skip_connect.append(sc_feat)
            pass

        out_feat = skip_connect[0]
        for skip in skip_connect[1:]:
            out_feat = out_feat + skip
        logits = self.readout_mlp(out_feat).view(-1)
        return skip_connect, out_feat, logits, torch.sigmoid(logits)

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv1 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        pass

    def forward(self, x, x2=None, x_gcn=None):
        x_size = x.size()
        y1 = self.conv1(self.pool2(x))
        y2 = self.conv2(self.pool4(x))
        y3 = self.conv3(self.pool8(x))
        res = torch.add(x, F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True))
        res = self.relu(res)

        if self.need_x2:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)
            pass

        res = self.conv_sum(res)

        if self.need_fuse:
            res = torch.add(res, x2)
            res = self.conv_sum_c(res)
            pass
        return res

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super(MyGCNNet, self).__init__()
        # BASE
        backbone = resnet.__dict__["resnet50"](pretrained=False, replace_stride_with_dilation=[False, False, True])
        return_layers = {'relu': 'e0', 'layer1': 'e1', 'layer2': 'e2', 'layer3': 'e3', 'layer4': 'e4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        for param in self.backbone.named_parameters():
            if "bn" in param[0]:
                param[1].requires_grad = False
            pass

        # Convert
        self.relu = nn.ReLU(inplace=True)
        self.convert5 = nn.Conv2d(2048, 512, 1, 1, bias=False)  # 25
        self.convert4 = nn.Conv2d(1024, 512, 1, 1, bias=False)  # 25
        self.convert3 = nn.Conv2d(512, 256, 1, 1, bias=False)  # 50
        self.convert2 = nn.Conv2d(256, 256, 1, 1, bias=False)  # 100
        self.convert1 = nn.Conv2d(64, 128, 1, 1, bias=False)  # 200

        # DEEP POOL
        deep_pool = [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128]]
        self.deep_pool5 = DeepPoolLayer(deep_pool[0][0], deep_pool[1][0], True, True)
        self.deep_pool4 = DeepPoolLayer(deep_pool[0][1], deep_pool[1][1], True, True)
        self.deep_pool3 = DeepPoolLayer(deep_pool[0][2], deep_pool[1][2], True, True)
        self.deep_pool2 = DeepPoolLayer(deep_pool[0][3], deep_pool[1][3], True, True)
        self.deep_pool1 = DeepPoolLayer(deep_pool[0][4], deep_pool[1][4], False, False)

        # ScoreLayer
        score = 128
        self.score = nn.Conv2d(score, 1, 1, 1)

        self.weight_init(self.modules())
        pass

    def forward(self, x):
        # BASE
        feature = self.backbone(x)
        feature1 = self.relu(self.convert1(feature["e0"]))  # 128, 200
        feature2 = self.relu(self.convert2(feature["e1"]))  # 256, 100
        feature3 = self.relu(self.convert3(feature["e2"]))  # 256, 50
        feature4 = self.relu(self.convert4(feature["e3"]))  # 512, 25
        feature5 = self.relu(self.convert5(feature["e4"]))  # 512, 25

        # SIZE
        x_size = x.size()[2:]

        merge = self.deep_pool5(feature5, feature4)  # A + F
        merge = self.deep_pool4(merge, feature3)  # A + F
        merge = self.deep_pool3(merge, feature2)  # A + F
        merge = self.deep_pool2(merge, feature1)  # A + F
        merge = self.deep_pool1(merge)  # A

        # ScoreLayer
        merge = self.score(merge)
        if x_size is not None:
            merge = F.interpolate(merge, x_size, mode='bilinear', align_corners=True)
        return merge, torch.sigmoid(merge)

    def load_pretrained_model(self, pretrained_model="./pretrained/resnet50-19c8e357.pth"):
        self.backbone.load_state_dict(torch.load(pretrained_model), strict=False)
        pass

    @staticmethod
    def weight_init(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            pass
        pass

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path, down_ratio=4, sp_size=4, train_print_freq=100, test_print_freq=50,
                 root_ckpt_dir="./ckpt2/norm3", lr=None, num_workers=8, use_gpu=True, gpu_id="1",
                 weight_decay=0.0, is_sgd=False):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(
            data_root_path=data_root_path, is_train=True, down_ratio=down_ratio, sp_size=sp_size)
        self.test_dataset = MyDataset(
            data_root_path=data_root_path, is_train=False, down_ratio=down_ratio, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet().to(self.device)
        self.model.eval()
        self.model.load_pretrained_model()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if is_sgd:
            self.lr_s = [[0, 0.01], [50, 0.001], [90, 0.0001]] if lr is None else lr
            self.optimizer = torch.optim.SGD(parameters, lr=self.lr_s[0][1], momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]] if lr is None else lr
            self.optimizer = torch.optim.Adam(parameters, lr=self.lr_s[0][1], weight_decay=weight_decay)

        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))
        self._print_network(self.model)

        self.loss_class = nn.BCELoss().to(self.device)
        pass

    def loss_bce(self, logits_sigmoid, labels):
        loss = self.loss_class(logits_sigmoid, labels)
        return loss

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_loss, train_mae, train_score = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            test_loss, test_mae, test_score = self.test()

            Tools.print('E:{:2d}, Train sod-mae-score={:.4f}-{:.4f} loss={:.4f}'.format(epoch, train_mae,
                                                                                        train_score, train_loss))
            Tools.print('E:{:2d}, Test  sod-mae-score={:.4f}-{:.4f} loss={:.4f}'.format(epoch, test_mae,
                                                                                        test_score, test_loss))
            pass
        pass

    def _train_epoch(self):
        self.model.eval()

        # 统计
        th_num = 25
        epoch_loss, nb_data = 0, 0
        epoch_mae, epoch_prec, epoch_recall = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        # Run
        iter_size = 10
        self.model.zero_grad()
        tr_num = len(self.train_loader)
        for i, (images, _, labels_sod, _, _) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)

            sod_logits, sod_logits_sigmoid = self.model.forward(images)

            loss_fuse1 = F.binary_cross_entropy_with_logits(sod_logits, labels_sod, reduction='sum')
            loss = loss_fuse1 / iter_size

            loss.backward()

            if (i + 1) % iter_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                pass

            labels_sod_val = labels_sod.cpu().detach().numpy()
            sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

            # Stat
            nb_data += images.size(0)
            epoch_loss += loss.detach().item()

            # cal 1
            mae = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
            prec, recall = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}) sod-mse={:.4f}({:.4f})".format(
                    i, tr_num, loss.detach().item(), epoch_loss / (i + 1), mae, epoch_mae / (i + 1)))
                pass
            pass

        # 结果
        avg_loss = epoch_loss / tr_num
        avg_mae, avg_prec, avg_recall = epoch_mae / tr_num, epoch_prec / tr_num, epoch_recall / tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)

        return avg_loss, avg_mae, score.max()

    def test(self, model_file=None, is_train_loader=False):
        self.model.eval()

        if model_file:
            self.load_model(model_file_name=model_file)

        Tools.print()
        th_num = 25

        # 统计
        epoch_test_loss, nb_data = 0, 0
        epoch_test_mae = 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        loader = self.train_loader if is_train_loader else self.test_loader
        tr_num = len(loader)
        with torch.no_grad():
            for i, (images, _, labels_sod, _, _) in enumerate(loader):
                # Data
                images = images.float().to(self.device)
                labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)

                _, sod_logits_sigmoid = self.model.forward(images)

                loss = self.loss_bce(sod_logits_sigmoid, labels_sod)

                labels_sod_val = labels_sod.cpu().detach().numpy()
                sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)
                epoch_test_loss += loss.detach().item()

                # cal 1
                mae = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
                prec, recall = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}) sod-mse={:.4f}({:.4f})".format(
                        i, len(loader), loss.detach().item(), epoch_test_loss/(i+1), mae, epoch_test_mae/(i+1)))
                    pass
                pass
            pass

        # 结果1
        avg_loss = epoch_test_loss/tr_num

        avg_mae, avg_prec, avg_recall = epoch_test_mae/tr_num, epoch_test_prec/tr_num, epoch_test_recall/tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)

        return avg_loss, avg_mae, score.max()

    @staticmethod
    def _print_network(model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        Tools.print(model)
        Tools.print("The number of parameters: {}".format(num_params))
        pass

    @staticmethod
    def _cal_sod(pre, segment):
        result = np.asarray(segment.copy(), dtype=np.float32)
        for i in range(len(pre)):
            result[segment == i] = pre[i]
            pass
        return result

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

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


"""
PYG_ResNet1_NoBN Total param: 51680449 lr_s=[[0, 5e-05], [20, 5e-06]]
2020-08-25 19:57:05 E:27, Train sod-mae-score=0.0115-0.9825 loss=267.9566
2020-08-25 19:57:05 E:27, Test  sod-mae-score=0.0398-0.8721 loss=0.1341

PYG_ResNet1_NoBN Total param: 51680449 lr_s=[[0, 1e-05], [20, 1e-06]]
2020-08-26 04:59:26 E:26, Train sod-mae-score=0.0111-0.9828 loss=253.5631
2020-08-26 04:59:26 E:26, Test  sod-mae-score=0.0385-0.8658 loss=0.1574
"""


if __name__ == '__main__':

    # _data_root_path = "/mnt/4T/Data/SOD/DUTS"
    # _data_root_path = "/media/ubuntu/data1/ALISURE/DUTS"
    _data_root_path = "/mnt/4T/ALISURE/DUTS"

    _train_print_freq = 1000
    _test_print_freq = 1000
    _num_workers = 10
    _use_gpu = True

    # _gpu_id = "0"
    # _gpu_id = "1"
    _gpu_id = "2"
    # _gpu_id = "3"

    _epochs = 30  # Super Param Group 1
    _is_sgd = False
    _weight_decay = 5e-4
    # _lr = [[0, 5e-05], [20, 5e-06]]
    _lr = [[0, 1e-5], [20, 1e-6]]

    _sp_size, _down_ratio = 4, 4

    _root_ckpt_dir = "./ckpt/PYG_ResNet1_NoBN/{}".format(_gpu_id)
    Tools.print("epochs:{} ckpt:{} sp size:{} down_ratio:{} workers:{} gpu:{} is_sgd:{} weight_decay:{}".format(
        _epochs, _root_ckpt_dir, _sp_size, _down_ratio, _num_workers, _gpu_id, _is_sgd, _weight_decay))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       sp_size=_sp_size, is_sgd=_is_sgd, lr=_lr, down_ratio=_down_ratio, weight_decay=_weight_decay,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs, start_epoch=0)
    pass
