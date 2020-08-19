import os
import cv2
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
from torchvision.models import vgg13_bn, vgg16_bn, resnet
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv


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

    def __init__(self, image_data, label_data, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        image_size = image_data.shape[0: 2]
        self.super_pixel_num = (image_size[0] * image_size[1]) // (super_pixel_size * super_pixel_size)
        self.image_data = image_data
        self.label_data = label_data
        try:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        except TypeError:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter)
            pass

        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        edge_index, sp_label, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            where = self.segment == i
            # 计算标签
            label = np.mean(self.label_data[where])
            sp_label.append(label)

            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(where, selem=skimage.morphology.square(3))
            edge_index.extend([[i, sp_id] for sp_id in np.unique(self.segment[_now_adj]) if sp_id != i])

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            pixel_edge_index = np.argwhere(_dis <= 2)
            pixel_edge_w = np.ones(len(pixel_edge_index))
            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w, label])
            pass

        sp_adj = np.asarray(edge_index)
        sp_label = np.asarray(sp_label)
        return self.segment, sp_adj, pixel_adj, sp_label

    pass


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


class MyDataset1(Dataset):

    def __init__(self, data_root_path, down_ratio=4, down_ratio2=2, is_train=True, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.down_ratio_for_sp = down_ratio
        self.down_ratio_for_sod = down_ratio2
        self.data_root_path = data_root_path

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
            graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_for_sp)
        else:
            Tools.print('IMAGE ERROR, PASSING {}'.format(image_name))
            graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name = \
                self.__getitem__(np.random.randint(0, len(self.image_name_list)))
            pass
        # 返回
        return graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name

    def get_sp_info(self, image, label):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=image, label_data=label, super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj, sp_label = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = Data(edge_index=torch.from_numpy(np.transpose(sp_adj, axes=(1, 0))),
                     num_nodes=len(pixel_adj), y=torch.from_numpy(sp_label).float(), num_sp=len(pixel_adj))
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([super_pixel[3]]),
                               edge_w=torch.from_numpy(super_pixel[2]).unsqueeze(1).float())
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph, segment

    @staticmethod
    def collate_fn(samples):
        graphs, pixel_graphs, images, labels_sp, labels_sod, segments, images_small, image_name = map(list,
                                                                                                      zip(*samples))

        images = torch.cat(images)
        images_small = torch.tensor(images_small)

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

        return images, labels_sp, labels_sod, batched_graph, batched_pixel_graph, segments, images_small, image_name

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
            img_data, label_for_sp, label_for_sod, image_small_data, image_name = \
                self.__getitem__(np.random.randint(0, len(self.image_name_list)))
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


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [8, 15, 22, 29]  # [3, 8, 15, 22, 29]
        self.features = self.vgg(self.cfg)

        self.weight_init(self.modules())
        pass

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.features)):
            x = self.features[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

    @staticmethod
    def vgg(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                # layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            pass
        return nn.Sequential(*layers)

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

    def load_pretrained_model(self, pretrained_model="./pretrained/vgg16-397923af.pth"):
        self.load_state_dict(torch.load(pretrained_model), strict=False)
        pass

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, is_not_last):
        super(DeepPoolLayer, self).__init__()
        self.is_not_last = is_not_last

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv1 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.is_not_last:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, x, x2=None):
        x_size = x.size()

        y1 = self.conv1(self.pool2(x))
        y2 = self.conv2(self.pool4(x))
        y3 = self.conv3(self.pool8(x))
        res = torch.add(x, F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True))
        res = self.relu(res)

        if self.is_not_last:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)

        res = self.conv_sum(res)

        if self.is_not_last:
            res = self.conv_sum_c(torch.add(res, x2))
        return res

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super(MyGCNNet, self).__init__()
        # BASE
        self.vgg16 = VGG16()

        # DEEP POOL
        deep_pool = [[512, 512, 256, 128], [512, 256, 128, 128]]
        self.deep_pool4 = DeepPoolLayer(deep_pool[0][0], deep_pool[1][0], True)
        self.deep_pool3 = DeepPoolLayer(deep_pool[0][1], deep_pool[1][1], True)
        self.deep_pool2 = DeepPoolLayer(deep_pool[0][2], deep_pool[1][2], True)
        self.deep_pool1 = DeepPoolLayer(deep_pool[0][3], deep_pool[1][3], False)

        # ScoreLayer
        score = 128
        self.score = nn.Conv2d(score, 1, 1, 1)

        VGG16.weight_init(self.modules())
        pass

    def forward(self, x):
        # BASE
        feature1, feature2, feature3, feature4 = self.vgg16(x)

        x_size = x.size()[2:]

        # DEEP POOL
        merge = self.deep_pool4(feature4, feature3)  # A + F
        merge = self.deep_pool3(merge, feature2)  # A + F
        merge = self.deep_pool2(merge, feature1)  # A + F
        merge = self.deep_pool1(merge)  # A

        # ScoreLayer
        merge = self.score(merge)
        if x_size is not None:
            merge = F.interpolate(merge, x_size, mode='bilinear', align_corners=True)
        return merge, torch.sigmoid(merge)

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
        self.model.vgg16.load_pretrained_model(
            pretrained_model="/mnt/4T/ALISURE/SOD/PoolNet/pretrained/vgg16-397923af.pth")

        if is_sgd:
            # self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]]
            self.lr_s = [[0, 0.01], [50, 0.001], [90, 0.0001]] if lr is None else lr
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][1],
                                             momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]] if lr is None else lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=weight_decay)

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

        # keys = [c for c in ckpt if "model_gnn1.gcn_list.0" in c]
        # for c in keys:
        #     del ckpt[c]
        #     Tools.print(c)
        #     pass

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
        self.model.train()

        # 统计
        th_num = 25
        epoch_loss, nb_data = 0, 0
        epoch_mae, epoch_prec, epoch_recall = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        # Run
        iter_size = 10
        self.model.zero_grad()
        for i, (images, labels_sp, labels_sod, _, _) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)

            sod_logits, sod_logits_sigmoid = self.model.forward(images)

            loss_fuse = F.binary_cross_entropy_with_logits(sod_logits, labels_sod, reduction='sum')
            loss = loss_fuse / iter_size

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
                    i, len(self.train_loader), loss.detach().item(), epoch_loss / (i + 1), mae, epoch_mae / (i + 1)))
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_loss / len(self.train_loader), epoch_mae / len(self.train_loader)
        avg_prec, avg_recall = epoch_prec / len(self.train_loader), epoch_recall / len(self.train_loader)
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)

        return avg_loss, avg_mae, score.max()

    def test(self, model_file=None, is_train_loader=False):
        if model_file:
            self.load_model(model_file_name=model_file)

        self.model.train()

        Tools.print()
        th_num = 25

        # 统计
        epoch_test_loss, nb_data = 0, 0
        epoch_test_mae = 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        loader = self.train_loader if is_train_loader else self.test_loader
        with torch.no_grad():
            for i, (images, labels_sp, labels_sod, _, _) in enumerate(loader):
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
        avg_loss, avg_mae = epoch_test_loss / len(loader), epoch_test_mae / len(loader)
        avg_prec, avg_recall = epoch_test_prec / len(loader), epoch_test_recall / len(loader)
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
# PoolNet - Info
2020-08-19 06:31:14 E:29, Train sod-mae-score=0.0091-0.9860 loss=215.3468
2020-08-19 06:31:14 E:29, Test  sod-mae-score=0.0392-0.8720 loss=0.1676
"""


if __name__ == '__main__':

    _data_root_path = "/mnt/4T/Data/SOD/DUTS"

    _train_print_freq = 1000
    _test_print_freq = 1000
    _num_workers = 16
    _use_gpu = True

    _gpu_id = "0"
    # _gpu_id = "1"

    _epochs = 30  # Super Param Group 1
    _is_sgd = False
    _weight_decay = 5e-4
    _lr = [[0, 5e-5], [20, 5e-6]]

    _improved = True
    _has_bn = True
    _has_residual = True
    _is_normalize = True
    _concat = True

    _sp_size, _down_ratio, _model_name = 4, 4, "C2PC2PC3C3C3"
    _name = "E2E2-Pretrain_temp3_BS1-MoreConv-{}_{}_lr0001".format(_model_name, _is_sgd)

    _root_ckpt_dir = "./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS_Temp/{}".format(_name)
    Tools.print("name:{} epochs:{} ckpt:{} sp size:{} down_ratio:{} workers:{} gpu:{} "
                "has_residual:{} is_normalize:{} has_bn:{} improved:{} concat:{} is_sgd:{} weight_decay:{}".format(
        _name, _epochs, _root_ckpt_dir, _sp_size, _down_ratio, _num_workers, _gpu_id,
        _has_residual, _is_normalize, _has_bn, _improved, _concat, _is_sgd, _weight_decay))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       sp_size=_sp_size, is_sgd=_is_sgd, lr=_lr,
                       down_ratio=_down_ratio, weight_decay=_weight_decay,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs, start_epoch=0)
    pass
