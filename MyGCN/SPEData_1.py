import os
import cv2
import torch
import numpy as np
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Dataset, Data
from visual_embedding_norm import DealSuperPixel, MyCIFAR10, Runner, EmbeddingNetCIFARSmallNorm3


class CIFAR10Graph(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        pass

    # __init__()中调用该函数
    def process(self):
        """对原始数据进行处理"""
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1
            pass
        pass

    def len(self):
        return len(self.processed_file_names)

    # __getitem__()中调用该函数
    def get(self, idx):
        """价值数据"""
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    pass


class MNISTSuperpixels(Dataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(MNISTSuperpixels, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        pass

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            x, edge_index, edge_slice, pos, y = torch.load(raw_path)
            edge_index, y = edge_index.to(torch.long), y.to(torch.long)
            m, n = y.size(0), 75
            x, pos = x.view(m * n, 1), pos.view(m * n, 2)
            node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
            graph_slice = torch.arange(m + 1, dtype=torch.long)
            self.data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            self.slices = {
                'x': node_slice,
                'edge_index': edge_slice,
                'y': graph_slice,
                'pos': node_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)
            pass
        pass

    pass


if __name__ == '__main__':

    # 1. Data
    data_root_path = 'D:\data\CIFAR'
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    train_set = MyCIFAR10(root=data_root_path, train=True, download=True, transform=transform_train)
    img, target = train_set.__getitem__(0)

    # 2. Model
    use_gpu = False
    gpu_id = "0"
    device = Runner.gpu_setup(use_gpu, gpu_id)
    model_file_name = "ckpt\\norm3\\epoch_1.pkl"
    model = EmbeddingNetCIFARSmallNorm3
    model = model().to(device)
    model.load_state_dict(torch.load(model_file_name), strict=False)

    # 3. Super Pixel
    ds_image_size = 32
    sp_size = 4
    sp_data_size = 6
    now_data_list, now_shape_list = [], []
    deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=ds_image_size, super_pixel_size=sp_size)
    now_segment, now_super_pixel_info, now_adjacency_info = deal_super_pixel.run()
    now_node_num = len(now_super_pixel_info)
    for key in now_super_pixel_info:
        _now_data = now_super_pixel_info[key]["data2"] / 255
        _now_data = cv2.resize(_now_data, (sp_data_size, sp_data_size), interpolation=cv2.INTER_NEAREST)
        _now_shape = now_super_pixel_info[key]["label"] / 1
        _now_shape = cv2.resize(_now_shape, (sp_data_size, sp_data_size), interpolation=cv2.INTER_NEAREST)
        now_data_list.append(_now_data)
        now_shape_list.append(np.expand_dims(_now_shape, axis=-1))
        pass
    net_data = np.transpose(now_data_list, axes=(0, 3, 1, 2))
    net_shape = np.transpose(now_shape_list, axes=(0, 3, 1, 2))
    sp_result = {"segment": now_segment, "node": now_super_pixel_info, "edge": now_adjacency_info}

    # 4. Visual Embedding
    net_data_tensor = torch.from_numpy(net_data).float().to(device)
    shape_feature, texture_feature, _, _ = model.forward(net_data_tensor)
    shape_feature, texture_feature = shape_feature.detach().numpy(), texture_feature.detach().numpy()
    for sp_i in range(now_node_num):
        sp_result["node"][sp_i]["feature_shape"] = shape_feature[sp_i]
        sp_result["node"][sp_i]["feature_texture"] = texture_feature[sp_i]
        pass

    pass