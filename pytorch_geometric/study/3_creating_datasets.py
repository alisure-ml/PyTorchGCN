import os
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset


class MyOwnDataset1(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        pass

    # raw_paths()中调用该函数
    @property
    def raw_file_names(self):  # 处理前的文件名
        return ['some_file_1', 'some_file_2', ...]

    # processed_paths()中调用该函数
    @property
    def processed_file_names(self):  # 处理后的文件名
        return ['data.pt']

    # __init__()中调用该函数
    def download(self):
        # Download to `self.raw_dir`.
        pass

    # __init__()中调用该函数
    def process(self):
        """对原始数据进行处理"""

        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            pass

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            pass

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        pass

    pass


class MyOwnDataset2(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        pass

    # raw_paths()中调用该函数
    @property
    def raw_file_names(self):  # 处理前的文件名
        return ['some_file_1', 'some_file_2', ...]

    # processed_paths()中调用该函数
    @property
    def processed_file_names(self):  # 处理后的文件名
        return ['data_1.pt', 'data_2.pt', ...]

    # __init__()中调用该函数
    def download(self):
        # Download to `self.raw_dir`.
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

