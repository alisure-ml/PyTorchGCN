import os
import cv2
import dgl
import glob
import time
import torch
import skimage
import numpy as np
from PIL import Image
import torch.nn as nn
from skimage import io
import matplotlib.pyplot as plt
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

        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                         sigma=slic_sigma, max_iter=slic_max_iter)
        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        sp_label, edge_index, pixel_adj = [], [], []
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
            pixel_adj.append([pixel_data_where, pixel_edge_index])
            pass

        sp_adj = np.asarray(edge_index)
        sp_label = np.asarray(sp_label)
        return self.segment, sp_adj, pixel_adj, sp_label

    pass


# class RandomScaleCrop(object):
#     def __init__(self, base_size, crop_size, fill=0):
#         self.base_size = base_size
#         self.crop_size = crop_size
#         self.fill = fill
#
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         # random scale (short edge)
#         short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         w, h = img.size
#         if h > w:
#             ow = short_size
#             oh = int(1.0 * h * ow / w)
#         else:
#             oh = short_size
#             ow = int(1.0 * w * oh / h)
#         img = img.resize((ow, oh), Image.BILINEAR)
#         mask = mask.resize((ow, oh), Image.NEAREST)
#         # pad crop
#         if short_size < self.crop_size:
#             padh = self.crop_size - oh if oh < self.crop_size else 0
#             padw = self.crop_size - ow if ow < self.crop_size else 0
#             img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
#             mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
#         # random crop crop_size
#         w, h = img.size
#         x1 = random.randint(0, w - self.crop_size)
#         y1 = random.randint(0, h - self.crop_size)
#         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#
#         return {'image': img, 'label': mask}


class MyDataset(Dataset):

    def __init__(self, data_root_path, is_train=True, image_size=320, sp_size=4, pool_ratio=4):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.pool_ratio = pool_ratio
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // pool_ratio
        self.data_image_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Image" if self.is_train else "DUTS-TE-Image")
        self.data_label_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Mask" if self.is_train else "DUTS-TE-Mask")

        self.transform_train = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])
        self.transform_test = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])
        self.transform_train_target = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])
        self.transform_test_target = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])

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
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image = self.transform_train(image) if self.is_train else self.transform_test(image)
        target = Image.open(self.label_name_list[idx])
        target = self.transform_train_target(target) if self.is_train else self.transform_test_target(target)

        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

        image_small_data = np.asarray(image.resize((self.image_size_for_sp, self.image_size_for_sp)))
        label_small_data = np.asarray(target.resize((self.image_size_for_sp, self.image_size_for_sp))) / 255
        graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_small_data)
        return graph, pixel_graph, img_data, target, segment

    def get_sp_info(self, image, label):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=image, label_data=label,
                                          ds_image_size=self.image_size_for_sp, super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj, sp_label = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = dgl.DGLGraph()
        graph.add_nodes(len(pixel_adj))
        graph.ndata['label'] = torch.from_numpy(sp_label).float()
        graph.add_edges(sp_adj[:, 0], sp_adj[:, 1])
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = dgl.DGLGraph()
            small_graph.add_nodes(len(super_pixel[0]))
            small_graph.ndata['data_where'] = torch.from_numpy(super_pixel[0]).long()
            small_graph.add_edges(super_pixel[1][:, 0], super_pixel[1][:, 1])
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph, segment

    @staticmethod
    def collate_fn(samples):
        graphs, pixel_graphs, images, targets, segments = map(list, zip(*samples))
        images = torch.cat(images)

        # 超像素图
        _nodes_num = [graph.number_of_nodes() for graph in graphs]
        nodes_num_norm = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        batch_graph = dgl.batch(graphs)

        # 像素图
        _pixel_graphs = []
        for super_pixel_i, pixel_graph in enumerate(pixel_graphs):
            for now_graph in pixel_graph:
                now_graph.ndata["data_where"][:, 0] = super_pixel_i
                _pixel_graphs.append(now_graph)
            pass
        _nodes_num = [graph.number_of_nodes() for graph in _pixel_graphs]
        pixel_nodes_num_norm = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        batch_pixel_graph = dgl.batch(_pixel_graphs)

        return images, batch_graph, nodes_num_norm, batch_pixel_graph, pixel_nodes_num_norm, targets, segments

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


class CONVNet(nn.Module):

    def __init__(self, in_dim, hidden_dims):
        super().__init__()

        layers = []
        for index, hidden_dim in enumerate(hidden_dims):
            if hidden_dim == "M":
                layers.append(nn.MaxPool2d((2, 2)))
            else:
                layers.append(ConvBlock(in_dim, int(hidden_dim), 1, padding=1, ks=3, has_bn=True))
                in_dim = int(hidden_dim)
            pass
        self.features = nn.Sequential(*layers)
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


class GCNNet1(nn.Module):

    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        self.gcn_list = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GCNLayer(in_dim, hidden_dim, F.relu, 0.0, True, True, True))
            in_dim = hidden_dim
        pass

    def forward(self, graphs, nodes_feat, nodes_num_norm_sqrt):
        hidden_nodes_feat = nodes_feat
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        return hg

    pass


class GCNNet2(nn.Module):

    def __init__(self, in_dim, hidden_dims, n_out=1):
        super().__init__()
        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.gcn_list = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GCNLayer(in_dim, hidden_dim, F.relu, 0.0, True, True, True))
            in_dim = hidden_dim
            pass
        self.readout = nn.Linear(in_dim, n_out, bias=False)
        pass

    def forward(self, graphs, nodes_feat, nodes_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        logits = self.readout(hidden_nodes_feat)
        logits = logits.view(-1)
        return logits, torch.sigmoid(logits)

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_conv = CONVNet(in_dim=3, hidden_dims=["64", "64", "M", "128", "128"])
        self.model_gnn1 = GCNNet1(in_dim=128, hidden_dims=[146, 146])
        self.model_gnn2 = GCNNet2(in_dim=146, hidden_dims=[146, 146, 146, 146], n_out=1)
        pass

    def forward(self, images, batched_graph, nodes_num_norm_sqrt,
                pixel_data_where, batched_pixel_graph, pixel_nodes_num_norm_sqrt):
        # model 1
        conv_feature = self.model_conv(images) if self.model_conv else images

        # model 2
        pixel_nodes_feat = conv_feature[pixel_data_where[:, 0], :, pixel_data_where[:, 1], pixel_data_where[:, 2]]
        batched_pixel_graph.ndata['feat'] = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph, pixel_nodes_feat, pixel_nodes_num_norm_sqrt)

        # model 3
        batched_graph.ndata['feat'] = gcn1_feature
        logits, logits_sigmoid = self.model_gnn2.forward(batched_graph, gcn1_feature, nodes_num_norm_sqrt)
        return logits, logits_sigmoid

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path, batch_size=64, image_size=320, sp_size=4, pool_ratio=2, train_print_freq=100,
                 test_print_freq=50, root_ckpt_dir="./ckpt2", num_workers=8, use_gpu=True, gpu_id="1"):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True,
                                       image_size=image_size, sp_size=sp_size, pool_ratio=pool_ratio)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False,
                                      image_size=image_size, sp_size=sp_size, pool_ratio=pool_ratio)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet().to(self.device)

        self.lr_s = [[0, 0.001], [25, 0.001], [50, 0.0003], [75, 0.0001]]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)
        # self.lr_s = [[0, 0.01], [50, 0.001], [80, 0.0001]]
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][0], momentum=0.9, weight_decay=5e-4)

        self.loss_class = nn.BCELoss().to(self.device)

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
            epoch_loss, epoch_train_mae, epoch_train_score = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_mae, epoch_test_score = self.test()

            Tools.print('E:{:02d}, lr:{:.4f}, Train(mae-score-loss):{:.4f}/{:.4f}/{:.4f} Test(mae-score-loss):'
                        '{:.4f}/{:.4f}/{:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr'],
                                                      epoch_train_mae, epoch_train_score, epoch_loss,
                                                      epoch_test_mae, epoch_test_score, epoch_test_loss))
            pass
        pass

    def _train_epoch(self):
        self.model.train()
        th_num = 100
        epoch_loss, nb_data = 0, 0
        epoch_mae = 0.0
        epoch_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall = np.zeros(shape=(th_num,)) + 1e-6
        for i, (images, batched_graph, nodes_num_norm_sqrt, batched_pixel_graph,
                pixel_nodes_num_norm_sqrt, targets, segments) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels = batched_graph.ndata["label"].to(self.device)
            nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
            pixel_data_where = batched_pixel_graph.ndata["data_where"].to(self.device)
            pixel_nodes_num_norm_sqrt = pixel_nodes_num_norm_sqrt.to(self.device)

            # Run
            self.optimizer.zero_grad()
            logits, logits_sigmoid = self.model.forward(images, batched_graph,
                                                        nodes_num_norm_sqrt, pixel_data_where,
                                                        batched_pixel_graph, pixel_nodes_num_norm_sqrt)
            loss = self.loss_class(logits_sigmoid, labels)
            labels_val = labels.cpu().detach().numpy()
            logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()
            loss.backward()
            self.optimizer.step()

            # Stat
            nb_data += labels.size(0)
            epoch_loss += loss.detach().item()

            mae = self._eval_mae(logits_sigmoid_val, labels_val)
            prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)

            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}/{:4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_mae/(i+1), mae))
                pass
            pass

        # 结果
        avg_loss, avg_mae = epoch_loss / len(self.train_loader), epoch_mae / len(self.train_loader)
        _avg_prec, _avg_recall = epoch_prec / len(self.train_loader), epoch_recall / len(self.train_loader)
        score = (1 + 0.3) * _avg_prec * _avg_recall / (0.3 * _avg_prec + _avg_recall)
        return avg_loss, avg_mae, score.max()

    def test(self):
        self.model.eval()

        Tools.print()
        th_num = 100
        epoch_test_loss, nb_data = 0, 0
        epoch_test_mae = 0.0
        epoch_test_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6
        with torch.no_grad():
            for i, (images, batched_graph, nodes_num_norm_sqrt, batched_pixel_graph,
                    pixel_nodes_num_norm_sqrt, targets, segments) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                labels = batched_graph.ndata["label"].to(self.device)
                nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
                pixel_data_where = batched_pixel_graph.ndata["data_where"].to(self.device)
                pixel_nodes_num_norm_sqrt = pixel_nodes_num_norm_sqrt.to(self.device)

                # Run
                logits, logits_sigmoid = self.model.forward(images, batched_graph,
                                                            nodes_num_norm_sqrt, pixel_data_where,
                                                            batched_pixel_graph, pixel_nodes_num_norm_sqrt)
                loss = self.loss_class(logits_sigmoid, labels)
                labels_val = labels.cpu().detach().numpy()
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += labels.size(0)
                epoch_test_loss += loss.detach().item()

                mae = self._eval_mae(logits_sigmoid_val, labels_val)
                prec, recall = self._eval_pr(logits_sigmoid_val, labels_val, th_num)

                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}/{:4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1),
                        loss.detach().item(), epoch_test_mae/(i+1), mae))
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

        def cal_sod(pre, segment):
            result = np.asarray(segment.copy(), dtype=np.float32)
            for i in range(len(pre)):
                result[segment == i] = pre[i]
                pass
            return result

        self.model.eval()
        with torch.no_grad():
            for i, (images, batched_graph, nodes_num_norm_sqrt, batched_pixel_graph,
                    pixel_nodes_num_norm_sqrt, targets, segments) in enumerate(self.test_loader):
                # Data
                images = images.float().to(self.device)
                nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
                pixel_data_where = batched_pixel_graph.ndata["data_where"].to(self.device)
                pixel_nodes_num_norm_sqrt = pixel_nodes_num_norm_sqrt.to(self.device)

                # Run
                logits, logits_sigmoid = self.model.forward(images, batched_graph,
                                                            nodes_num_norm_sqrt, pixel_data_where,
                                                            batched_pixel_graph, pixel_nodes_num_norm_sqrt)

                # 可视化
                labels = batched_graph.ndata["label"]
                logits_sigmoid_val = logits_sigmoid.cpu().detach().numpy()
                cum_add = np.cumsum([0] + batched_graph.batch_num_nodes)
                for one in range(len(segments)):
                    tar_sod = cal_sod(labels[cum_add[one]: cum_add[one+1]].tolist(), segments[one])
                    pre_sod = cal_sod(logits_sigmoid_val[cum_add[one]: cum_add[one+1]].tolist(), segments[one])

                    Image.fromarray(np.asarray(tar_sod * 255, dtype=np.uint8)).show()
                    Image.fromarray(np.asarray(pre_sod * 255, dtype=np.uint8)).show()
                    targets[one].show()
                    pass
                pass
            pass
        pass

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
    数据增强、真正的测试
    """
    # _data_root_path = 'D:\\data\\SOD\\DUTS'
    # _root_ckpt_dir = "ckpt3\\dgl\\my\\{}".format("GCNNet")
    # _batch_size = 16
    # _image_size = 320
    # _sp_size = 4
    # _epochs = 100
    # _train_print_freq = 1
    # _test_print_freq = 1
    # _num_workers = 1
    # _use_gpu = False
    # _gpu_id = "1"

    # _data_root_path = '/mnt/4T/Data/cifar/cifar-10'
    _data_root_path = '/home/ubuntu/ALISURE/data/SOD/DUTS'
    _root_ckpt_dir = "./ckpt3/dgl/6_DGL_SOD/{}".format("GCNNet")
    _batch_size = 8
    _image_size = 320
    _sp_size = 4
    _epochs = 100
    _train_print_freq = 100
    _test_print_freq = 50
    _num_workers = 8
    _use_gpu = True
    _gpu_id = "0"
    # _gpu_id = "1"

    Tools.print("ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    # runner.visual(model_file="./ckpt3/dgl/6_DGL_SOD/GCNNet-100/epoch_48.pkl")
    runner.train(_epochs)

    pass
