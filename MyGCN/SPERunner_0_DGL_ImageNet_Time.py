import os
import cv2
import dgl
import glob
import time
import torch
import shutil
import pickle
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
from torchvision.models import vgg13_bn, vgg16_bn


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

    def __init__(self, image_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_num = (self.ds_image_size // super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        # self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
        #                                  sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                         sigma=slic_sigma, max_iter=slic_max_iter)
        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        edge_index, edge_w, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(self.segment == i, selem=skimage.morphology.square(3))

            _adj_dis = []
            for sp_id in np.unique(self.segment[_now_adj]):
                if sp_id != i:
                    edge_index.append([i, sp_id])
                    _adj_dis.append(1/(np.sqrt((self.region_props[i][0][0] - self.region_props[sp_id][0][0]) ** 2 +
                                               (self.region_props[i][0][1] - self.region_props[sp_id][0][1]) ** 2) + 1))
                pass
            edge_w.extend(_adj_dis / np.sum(_adj_dis, axis=0))

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            _dis = _dis <= 2
            pixel_edge_index = np.argwhere(_dis)
            pixel_edge_w = np.ones(len(pixel_edge_index))
            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w])
            pass

        sp_adj = [np.asarray(edge_index), np.asarray(edge_w)]
        return self.segment, sp_adj, pixel_adj

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\\data\\ImageNet\\ILSVRC2015\\Data\\CLS-LOC',
                 is_train=True, image_size=224, sp_size=11, train_split="train", test_split="val"):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.image_size = image_size
        self.image_size_for_sp = self.image_size // 4
        self.data_root_path = data_root_path

        self.transform_train = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomCrop(self.image_size),
                                                   # transforms.RandomResizedCrop(self.image_size),
                                                   transforms.RandomHorizontalFlip()])
        self.transform_test = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(self.image_size)])

        _test_dir = os.path.join(self.data_root_path, test_split)
        _train_dir = os.path.join(self.data_root_path, train_split)
        _transform = self.transform_train if self.is_train else self.transform_test

        self.data_set = datasets.ImageFolder(root=_train_dir if self.is_train else _test_dir, transform=_transform)
        # self.data_set.samples = self.data_set.samples[0: 6]
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img, target = self.data_set.__getitem__(idx)
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(np.asarray(img)).unsqueeze(dim=0)

        img_small_data = np.asarray(img.resize((self.image_size_for_sp, self.image_size_for_sp)))
        graph, pixel_graph = self.get_sp_info(img_small_data)
        return graph, pixel_graph, img_data, target

    def get_sp_info(self, img):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(img, ds_image_size=self.image_size_for_sp, super_pixel_size=self.sp_size)
        segment, sp_adj, pixel_adj = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = dgl.DGLGraph()
        graph.add_nodes(len(pixel_adj))
        graph.add_edges(sp_adj[0][:, 0], sp_adj[0][:, 1])
        graph.edata['feat'] = torch.from_numpy(sp_adj[1]).unsqueeze(1).float()
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = dgl.DGLGraph()
            small_graph.add_nodes(len(super_pixel[0]))
            small_graph.ndata['data_where'] = torch.from_numpy(super_pixel[0]).long()
            small_graph.add_edges(super_pixel[1][:, 0], super_pixel[1][:, 1])
            small_graph.edata['feat'] = torch.from_numpy(super_pixel[2]).unsqueeze(1).float()
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph

    @staticmethod
    def collate_fn(samples):
        total_time = 0

        # 1 0.03
        graphs, pixel_graphs, images, labels = map(list, zip(*samples))
        images = torch.cat(images)
        labels = torch.tensor(np.array(labels))

        # 2 0.003 超像素图
        _nodes_num = [graph.number_of_nodes() for graph in graphs]
        _edges_num = [graph.number_of_edges() for graph in graphs]
        nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()

        # 3 0.02
        batched_graph = dgl.batch(graphs)

        start_time = time.time()

        # 4 0.62 像素图
        _pixel_graphs = []
        for super_pixel_i, pixel_graph in enumerate(pixel_graphs):
            for now_graph in pixel_graph:
                now_graph.ndata["data_where"][:, 0] = super_pixel_i
                _pixel_graphs.append(now_graph)
            pass
        _nodes_num = [graph.number_of_nodes() for graph in _pixel_graphs]
        _edges_num = [graph.number_of_edges() for graph in _pixel_graphs]
        pixel_nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        pixel_edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()

        # 5 0.78
        batched_pixel_graph = dgl.batch(_pixel_graphs)

        total_time += time.time() - start_time
        Tools.print("collate_fn ****************************** {}".format(total_time))

        # with open("./now2.pkl", "wb") as f:
        #     pickle.dump(batched_pixel_graph, f)

        return (images, labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt,
                batched_pixel_graph, pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt)

    pass


class RunnerSPE(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10',
                 batch_size=64, image_size=224, sp_size=8, train_print_freq=100, test_print_freq=50, is_sgd=True,
                 test_split="val", root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="1"):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False,
                                      image_size=image_size, sp_size=sp_size, test_split=test_split)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=4, collate_fn=self.test_dataset.collate_fn)
        pass

    def test(self):
        for i, (images, labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt, batched_pixel_graph,
                pixel_nodes_num_norm_sqrt, pixel_edges_num_norm_sqrt) in enumerate(self.test_loader):
            pass
        pass

    pass


if __name__ == '__main__':
    """
    GCNNet  2166696 32 sgd 0.01 2020-05-11 Epoch:03,lr=0.0100,Train:0.1530-0.3666/4.2644 Test:0.1470-0.3580/4.3808
    
    Load Model: ./ckpt2/dgl/4_DGL_CONV-ImageNet2/GCNNet2/epoch_3.pkl
    GCNNet3 4358696 36 sgd 0.01 2020-05-23 Epoch:09,lr=0.0001,Train:0.5551-0.7980/1.9078 Test:0.5512-0.7965/1.9085
    
    2020-06-02 21:45:23 Epoch:04, Train:0.2958-0.5602/3.3328 Test:0.2768-0.5402/3.5083
    """
    _data_root_path = '/mnt/4T/Data/ILSVRC17/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC'
    # _data_root_path = "/media/ubuntu/ALISURE-SSD/data/ImageNet/ILSVRC2015/Data/CLS-LOC"
    # _data_root_path = "/media/ubuntu/ALISURE/data/ImageNet/ILSVRC2015/Data/CLS-LOC"
    _root_ckpt_dir = "./ckpt2/dgl/4_DGL_CONV-ImageNet3_Fast/{}".format("GCNNet-C2PC2PC2")
    _batch_size = 64
    _image_size = 224
    _is_sgd = True
    _sp_size = 4
    _train_print_freq = 2000
    _test_print_freq = 1000
    _num_workers = 12
    _use_gpu = True
    _gpu_id = "0"
    # _gpu_id = "1"

    Tools.print("ckpt:{} batch size:{} image size:{} sp size:{} workers:{} gpu:{}".format(
        _root_ckpt_dir, _batch_size, _image_size, _sp_size, _num_workers, _gpu_id))

    runner = RunnerSPE(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir, test_split="val",
                       batch_size=_batch_size, image_size=_image_size, sp_size=_sp_size, is_sgd=_is_sgd,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    # runner.load_model(model_file_name="./ckpt2/dgl/4_DGL_CONV-ImageNet2/GCNNet4/epoch_3.pkl")
    # runner.train(10, start_epoch=0)
    runner.test()
    pass
