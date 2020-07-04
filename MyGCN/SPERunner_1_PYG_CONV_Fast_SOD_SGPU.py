import os
import cv2
import time
import glob
import torch
import skimage
import numpy as np
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg13_bn, vgg16_bn
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

    def __init__(self, image_data, label_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))
        self.label_data = label_data if len(label_data) == self.ds_image_size else cv2.resize(
            label_data, (self.ds_image_size, self.ds_image_size))

        # self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
        #                                  sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                         sigma=slic_sigma, max_iter=slic_max_iter)

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


class MyDataset(Dataset):

    def __init__(self, data_root_path, down_ratio=4, is_train=True, image_size=320, sp_size=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // down_ratio
        self.data_root_path = data_root_path

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
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

        # 超像素
        image_small_data = np.asarray(image.resize((self.image_size_for_sp, self.image_size_for_sp)))
        label_small_data = np.asarray(label.resize((self.image_size_for_sp, self.image_size_for_sp))) / 255
        graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_small_data)

        # 返回
        return graph, pixel_graph, img_data, label_small_data, segment

    def get_sp_info(self, image, label):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=image, label_data=label,
                                          ds_image_size=self.image_size_for_sp, super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj, sp_label = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = Data(edge_index=torch.from_numpy(np.transpose(sp_adj, axes=(1, 0))),
                     num_nodes=len(pixel_adj), y=torch.from_numpy(sp_label).float())
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
        graphs, pixel_graphs, images, labels, segments = map(list, zip(*samples))

        images = torch.cat(images)
        segments = torch.cat(segments)
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

        return images, labels, batched_graph, batched_pixel_graph, segments

    pass


class CONVNet(nn.Module):

    def __init__(self, layer_num=14):  # 14, 20
        super().__init__()
        self.features = vgg13_bn(pretrained=True).features[0: layer_num]
        pass

    def forward(self, x):
        e = self.features(x)
        return e
    pass


class GCNNet1(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.improved = improved

        # self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        # hidden_nodes_feat = self.embedding_h(data.x)
        hidden_nodes_feat = data.x
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

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], skip_which=[1, 2, 3], skip_dim=128, n_out=1,
                 has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.improved = improved

        self.embedding_h = nn.Linear(in_dim, in_dim)

        self.gcn_list = nn.ModuleList()
        _in_dim = in_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(GCNConv(_in_dim, hidden_dim, normalize=self.normalize, improved=self.improved))
            _in_dim = hidden_dim
            pass

        if self.has_bn:
            self.bn_list = nn.ModuleList()
            for hidden_dim in self.hidden_dims:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
                pass
            pass

        # skip
        self.skip_which = skip_which
        self.skip_dim = skip_dim
        self.n_out = n_out
        sk_hidden_dims = [in_dim] + [self.hidden_dims[which-1] for which in self.skip_which]
        self.skip_connect_index = [0] + self.skip_which
        self.skip_connect_list = nn.ModuleList()
        for hidden_dim in sk_hidden_dims:
            # 待改进
            self.skip_connect_list.append(nn.Linear(hidden_dim, self.skip_dim, bias=False))
            pass

        self.readout_mlp = nn.Linear(len(self.skip_connect_list) * skip_dim, n_out, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        gcn_hidden_nodes_feat = [hidden_nodes_feat]
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat
            hidden_nodes_feat = gcn(h_in, data.edge_index)

            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)

            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat

            gcn_hidden_nodes_feat.append(hidden_nodes_feat)
            pass

        skip_connect = []
        for sc, index in zip(self.skip_connect_list, self.skip_connect_index):
            sc_feat = sc(gcn_hidden_nodes_feat[index])
            skip_connect.append(sc_feat)
            pass

        out_feat = torch.cat(skip_connect, dim=1)
        logits = self.readout_mlp(out_feat).view(-1)
        return logits, torch.sigmoid(logits)

    pass


class MyGCNNet(nn.Module):

    def __init__(self, conv_layer_num=14, has_bn=False, normalize=False, residual=False, improved=False):
        super().__init__()
        self.model_conv = CONVNet(layer_num=conv_layer_num)  # 14, 20

        assert conv_layer_num == 14 or conv_layer_num == 20
        in_dim_which = -3 if conv_layer_num == 14 else -2
        self.model_gnn1 = GCNNet1(in_dim=self.model_conv.features[in_dim_which].num_features,
                                  hidden_dims=[256, 256],
                                  has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
        self.model_gnn2 = GCNNet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[512, 512, 1024, 1024],
                                  skip_which=[2, 4], skip_dim=128, n_out=1,
                                  has_bn=has_bn, normalize=normalize, residual=residual, improved=improved)
        pass

    def forward(self, images, batched_graph, batched_pixel_graph):
        # model 1
        conv_feature = self.model_conv(images)

        # model 2
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = conv_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph)

        # model 3
        batched_graph.x = gcn1_feature
        logits, logits_sigmoid = self.model_gnn2.forward(batched_graph)
        return logits, logits_sigmoid

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path, down_ratio=4, batch_size=64, image_size=320,
                 sp_size=4, train_print_freq=100, test_print_freq=50, root_ckpt_dir="./ckpt2/norm3",
                 num_workers=8, use_gpu=True, gpu_id="1", conv_layer_num=14,
                 has_bn=True, normalize=True, residual=False, improved=False, weight_decay=0.0, is_sgd=False):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True, down_ratio=down_ratio,
                                       image_size=image_size, sp_size=sp_size)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False, down_ratio=down_ratio,
                                      image_size=image_size, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(conv_layer_num=conv_layer_num,
                              has_bn=has_bn, normalize=normalize, residual=residual, improved=improved).to(self.device)

        if is_sgd:
            self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]]
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][1],
                                             momentum=0.9, weight_decay=weight_decay)
        else:
            self.lr_s = [[0, 0.001], [50, 0.0001], [90, 0.00001]]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=weight_decay)

        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))

        self.loss_class = nn.BCELoss().to(self.device)
        pass

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

            train_loss, train_mae, train_score, train_mae2, train_score2, train_mae3, train_score3 = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            test_loss, test_mae, test_score, test_mae2, test_score2, test_mae3, test_score3 = self.test()

            Tools.print('E:{:2d}, Train mae-score={:.4f}/{:.4f} '
                        'final-mse-score={:.4f}/{:.4f}-{:.4f}/{:.4f} loss={:.4f}'.format(
                epoch, train_mae, train_score, train_mae2, train_score2, train_mae3, train_score3, train_loss))
            Tools.print('E:{:2d}, Test  mae-score={:.4f}/{:.4f} '
                        'final-mse-score={:.4f}/{:.4f}-{:.4f}/{:.4f} loss={:.4f}'.format(
                epoch, test_mae, test_score, test_mae2, test_score2, test_mae3, test_score3, test_loss))
            pass
        pass

    def _train_epoch(self, is_print_result=False):
        self.model.train()
        th_num = 100

        # 统计
        epoch_loss, nb_data = 0, 0
        epoch_mae = 0.0
        epoch_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall = np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae2 = 0.0
        epoch_prec2 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall2 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae3 = 0.0
        epoch_prec3 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall3 = np.zeros(shape=(th_num,)) + 1e-6

        for i, (images, targets, batched_graph, batched_pixel_graph, segments) in enumerate(self.train_loader):
            # Run
            self.optimizer.zero_grad()

            # Data
            images = images.float().to(self.device)
            labels = batched_graph.y.to(self.device)
            batched_graph.batch = batched_graph.batch.to(self.device)
            batched_graph.edge_index = batched_graph.edge_index.to(self.device)

            batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
            batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
            batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

            logits, logits_sigmoid = self.model.forward(images, batched_graph, batched_pixel_graph)

            loss = self.loss_class(logits_sigmoid, labels)
            labels_val = labels.cpu().detach().numpy()
            logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()
            loss.backward()
            self.optimizer.step()

            # Stat
            nb_data += images.size(0)
            epoch_loss += loss.detach().item()

            # cal 1
            mae = self._eval_mae(logits_sigmoid_val, labels_val)
            prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # cal 2
            cum_add = np.cumsum([0] + batched_graph.batch_num_nodes)
            for one in range(len(segments)):
                tar_sod = self._cal_sod(labels[cum_add[one]: cum_add[one + 1]].tolist(), segments[one])
                pre_sod = self._cal_sod(logits_sigmoid_val[cum_add[one]: cum_add[one + 1]].tolist(), segments[one])

                mae2 = self._eval_mae(pre_sod, tar_sod)
                prec2, recall2 = self._eval_pr(pre_sod, tar_sod, th_num)
                epoch_mae2 += mae2
                epoch_prec2 += prec2
                epoch_recall2 += recall2

                mae3 = self._eval_mae(pre_sod, targets[one])
                prec3, recall3 = self._eval_pr(pre_sod, targets[one], th_num)
                epoch_mae3 += mae3
                epoch_prec3 += prec3
                epoch_recall3 += recall3
                pass

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{:4d}-{:4d} loss={:.4f}/{:.4f} mse={:.4f}/{:.4f} final-mse={:.4f}-{:.4f}".format(
                    i, len(self.train_loader), epoch_loss / (i + 1), loss.detach().item(),
                    mae, epoch_mae / (i + 1), epoch_mae2 / nb_data, epoch_mae3 / nb_data))
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_loss / len(self.train_loader), epoch_mae / len(self.train_loader)
        avg_prec, avg_recall = epoch_prec / len(self.train_loader), epoch_recall / len(self.train_loader)
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        # 结果2
        avg_mae2, avg_prec2, avg_recall2 = epoch_mae2/nb_data, epoch_prec2/nb_data, epoch_recall2/nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)
        # 结果3
        avg_mae3, avg_prec3, avg_recall3 = epoch_mae3/nb_data, epoch_prec3/nb_data, epoch_recall3/nb_data
        score3 = (1 + 0.3) * avg_prec3 * avg_recall3 / (0.3 * avg_prec3 + avg_recall3)

        if is_print_result:
            Tools.print('Train mae-score={:.4f}/{:.4f} final-mse-score={:.4f}/{:.4f}-{:.4f}/{:.4f} loss={:.4f}'.format(
                avg_mae, score.max(), avg_mae2, score2.max(), avg_mae3, score3.max(), avg_loss))
            pass

        return avg_loss, avg_mae, score.max(), avg_mae2, score2.max(), avg_mae3, score3.max()

    def test(self, model_file=None, is_print_result=False, is_train_loader=False):
        if model_file:
            self.load_model(model_file_name=model_file)

        self.model.eval()

        Tools.print()
        th_num = 100

        # 统计
        epoch_test_loss, nb_data = 0, 0
        epoch_test_mae = 0.0
        epoch_test_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_mae2 = 0.0
        epoch_test_prec2 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_recall2 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_mae3 = 0.0
        epoch_test_prec3 = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_recall3 = np.zeros(shape=(th_num,)) + 1e-6

        loader = self.train_loader if is_train_loader else self.test_loader
        with torch.no_grad():
            for i, (images, targets, batched_graph, batched_pixel_graph, segments) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)

                labels = batched_graph.y.to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                logits, logits_sigmoid = self.model.forward(images, batched_graph, batched_pixel_graph)
                loss = self.loss_class(logits_sigmoid, labels)
                labels_val = labels.cpu().detach().numpy()
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)
                epoch_test_loss += loss.detach().item()

                # cal 1
                mae = self._eval_mae(logits_sigmoid_val, labels_val)
                prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)
                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # cal 2
                cum_add = np.cumsum([0] + batched_graph.batch_num_nodes)
                for one in range(len(segments)):
                    tar_sod = self._cal_sod(labels[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_sod = self._cal_sod(logits_sigmoid_val[cum_add[one]: cum_add[one+1]].tolist(), segments[one])

                    mae2 = self._eval_mae(pre_sod, tar_sod)
                    prec2, recall2 = self._eval_pr(pre_sod, tar_sod, th_num)
                    epoch_test_mae2 += mae2
                    epoch_test_prec2 += prec2
                    epoch_test_recall2 += recall2

                    mae3 = self._eval_mae(pre_sod, targets[one])
                    prec3, recall3 = self._eval_pr(pre_sod, targets[one], th_num)
                    epoch_test_mae3 += mae3
                    epoch_test_prec3 += prec3
                    epoch_test_recall3 += recall3
                    pass

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{:4d}-{:4d} loss={:.4f}/{:.4f} mse={:.4f}/{:.4f} final-mse={:.4f}-{:.4f}".format(
                        i, len(loader), epoch_test_loss/(i+1), loss.detach().item(),
                        mae, epoch_test_mae/(i+1), epoch_test_mae2/nb_data, epoch_test_mae3/nb_data))
                    pass
                pass
            pass

        # 结果1
        avg_loss, avg_mae = epoch_test_loss / len(loader), epoch_test_mae / len(loader)
        avg_prec, avg_recall = epoch_test_prec / len(loader), epoch_test_recall / len(loader)
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        # 结果2
        avg_mae2, avg_prec2, avg_recall2 = epoch_test_mae2/nb_data, epoch_test_prec2/nb_data, epoch_test_recall2/nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)
        # 结果3
        avg_mae3, avg_prec3, avg_recall3 = epoch_test_mae3/nb_data, epoch_test_prec3/nb_data, epoch_test_recall3/nb_data
        score3 = (1 + 0.3) * avg_prec3 * avg_recall3 / (0.3 * avg_prec3 + avg_recall3)

        if is_print_result:
            Tools.print('Test mae-score={:.4f}/{:.4f} final-mse-score={:.4f}/{:.4f}-{:.4f}/{:.4f} loss={:.4f}'.format(
                avg_mae, score.max(), avg_mae2, score2.max(), avg_mae3, score3.max(), avg_loss))
            pass
        return avg_loss, avg_mae, score.max(), avg_mae2, score2.max(), avg_mae3, score3.max()

    def visual(self, model_file=None, is_train=False):
        if model_file:
            self.load_model(model_file_name=model_file)

        loader = self.train_loader if is_train else self.test_loader
        self.model.eval()
        with torch.no_grad():
            for i, (images, targets, batched_graph, batched_pixel_graph, segments) in enumerate(loader):
                # Data
                images = images.float().to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                # Run
                logits, logits_sigmoid = self.model.forward(images, batched_graph, batched_pixel_graph)

                # 可视化
                labels = batched_graph.y
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()
                cum_add = np.cumsum([0] + batched_graph.batch_num_nodes)
                for one in range(len(segments)):
                    tar_sod = self._cal_sod(labels[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_sod = self._cal_sod(logits_sigmoid_val[cum_add[one]: cum_add[one+1]].tolist(), segments[one])

                    Image.fromarray(np.asarray(tar_sod * 255, dtype=np.uint8)).show()
                    Image.fromarray(np.asarray(pre_sod * 255, dtype=np.uint8)).show()
                    # targets[one].show()
                    pass
                pass
            pass
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
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
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

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    """
    2020-07-04 15:34:16 Epoch:29, Train:0.5638-0.7989/1.9539 Test:0.5556-0.7971/1.9430
    """
    _data_root_path = '/home/ubuntu/ALISURE/data/SOD/DUTS'
    _root_ckpt_dir = "./ckpt2/dgl/1_PYG_CONV_Fast-SOD/{}".format("GCNNet-C2PC2P")
    _batch_size = 4
    _image_size = 320
    _train_print_freq = 100
    _test_print_freq = 100
    _num_workers = 16
    _use_gpu = True

    _gpu_id = "0"

    _epochs = 100
    _is_sgd = True
    _weight_decay = 5e-4

    _improved = True
    _has_bn = True
    _has_residual = True
    _is_normalize = True

    _sp_size, _down_ratio, _conv_layer_num = 4, 4, 14  # GCNNet-C2PC2P
    # _sp_size, _down_ratio, _conv_layer_num = 4, 4, 20  # GCNNet-C2PC2PC2

    Tools.print("epochs:{} ckpt:{} batch size:{} image size:{} sp size:{} down_ratio:{} conv_layer_num:{} workers:{} "
                "gpu:{} has_residual:{} is_normalize:{} has_bn:{} improved:{} is_sgd:{} weight_decay:{}".format(
        _epochs, _root_ckpt_dir, _batch_size, _image_size, _sp_size, _down_ratio, _conv_layer_num, _num_workers,
        _gpu_id, _has_residual, _is_normalize, _has_bn, _improved, _is_sgd, _weight_decay))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       residual=_has_residual, normalize=_is_normalize, down_ratio=_down_ratio,
                       has_bn=_has_bn, improved=_improved, weight_decay=_weight_decay, conv_layer_num=_conv_layer_num,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.load_model("./ckpt2/dgl/1_PYG_CONV_Fast-ImageNet/GCNNet-C2PC2P/epoch_29.pkl")
    runner.train(_epochs, start_epoch=0)

    pass
