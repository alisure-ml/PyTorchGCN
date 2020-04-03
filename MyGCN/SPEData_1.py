import os
import cv2
import torch
import numpy as np
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from visual_embedding_norm import DealSuperPixel, MyCIFAR10, Runner, EmbeddingNetCIFARSmallNorm3


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True, device=None,
                 ve_model_file_name="ckpt\\norm3\\epoch_1.pkl", VEModel=EmbeddingNetCIFARSmallNorm3,
                 image_size=32, sp_size=4, sp_ve_size=6):
        super().__init__()

        # 1. Data
        self.is_train = is_train
        self.data_root_path = data_root_path
        self.ve_transform = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
                                                transforms.RandomHorizontalFlip()]) if self.is_train else None
        self.ve_data_set = MyCIFAR10(root=self.data_root_path, train=self.is_train, transform=self.ve_transform)

        # 2. Model
        self.device = torch.device("cpu") if device is None else device
        self.ve_model_file_name = ve_model_file_name
        self.ve_model = VEModel(is_train=False).to(self.device)
        self.ve_model.load_state_dict(torch.load(self.ve_model_file_name), strict=False)

        # 3. Super Pixel
        self.image_size = image_size
        self.sp_size = sp_size
        self.sp_ve_size = sp_ve_size
        pass

    def __len__(self):
        return len(self.ve_data_set)

    def __getitem__(self, idx):
        img, target = self.ve_data_set.__getitem__(idx)
        g_data = self.get_sp_info(img, target)
        return g_data

    def get_sp_info(self, img, target):
        # 3. Super Pixel
        deal_super_pixel = DealSuperPixel(image_data=img, ds_image_size=self.image_size, super_pixel_size=self.sp_size)
        segment, super_pixel_info, adjacency_info = deal_super_pixel.run()

        # Resize Super Pixel
        _now_data_list = []
        for key in super_pixel_info:
            _now_data = cv2.resize(super_pixel_info[key]["data2"] / 255,
                                   (self.sp_ve_size, self.sp_ve_size), interpolation=cv2.INTER_NEAREST)
            _now_data_list.append(_now_data)
            pass
        net_data = np.transpose(_now_data_list, axes=(0, 3, 1, 2))

        # 4. Visual Embedding
        shape_feature, texture_feature = self.ve_model.forward(torch.from_numpy(net_data).float().to(self.device))
        shape_feature, texture_feature = shape_feature.detach().numpy(), texture_feature.detach().numpy()
        for sp_i in range(len(super_pixel_info)):
            super_pixel_info[sp_i]["feature_shape"] = shape_feature[sp_i]
            super_pixel_info[sp_i]["feature_texture"] = texture_feature[sp_i]
            pass

        # Data for Batch: super_pixel_info
        x, pos, area, size = [], [], [], []
        for sp_i in range(len(super_pixel_info)):
            now_sp = super_pixel_info[sp_i]

            _size = now_sp["size"]
            _area = now_sp["area"]
            _x = np.concatenate([now_sp["feature_shape"], now_sp["feature_texture"]], axis=0)

            x.append(_x)
            size.append([_size])
            area.append(_area)
            pos.append([_area[1] - _area[0], _area[3] - _area[2]])
            pass

        # Data for Batch: adjacency_info
        edge_index, edge_w = [], []
        for edge_i in range(len(adjacency_info)):
            edge_index.append([adjacency_info[edge_i][0], adjacency_info[edge_i][1]])
            edge_w.append([adjacency_info[edge_i][2]])
            pass
        edge_index = np.transpose(edge_index, axes=(1, 0))

        # Data for Batch: Data
        g_data = Data(x=torch.from_numpy(np.asarray(x)), edge_index=torch.from_numpy(edge_index),
                      y=torch.tensor([target]), pos=torch.from_numpy(np.asarray(pos)),
                      area=torch.from_numpy(np.asarray(area)), size=torch.from_numpy(np.asarray(size)),
                      edge_w=torch.from_numpy(np.asarray(edge_w)))
        return g_data

    @staticmethod
    def collate_fn(batch_data):
        return Batch.from_data_list(batch_data)

    pass


if __name__ == '__main__':

    _device = Runner.gpu_setup(use_gpu=False, gpu_id="0")

    my_dataset = MyDataset(data_root_path='D:\data\CIFAR', is_train=True, device=_device,
                           ve_model_file_name="ckpt\\norm3\\epoch_1.pkl", VEModel=EmbeddingNetCIFARSmallNorm3,
                           image_size=32, sp_size=4, sp_ve_size=6)

    data_loader = DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=my_dataset.collate_fn)

    for i, data in enumerate(data_loader):
        Tools.print("{}".format(i))
        pass

    pass
