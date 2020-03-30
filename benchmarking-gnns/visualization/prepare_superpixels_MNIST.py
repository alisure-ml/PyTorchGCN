import os
import time
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset
from data.superpixels import SuperPixDatasetDGL


start = time.time()
DATASET_NAME = 'MNIST'
now_data_dir = "D:\data\GCN\superpixels"
dataset = SuperPixDatasetDGL(DATASET_NAME, data_dir=now_data_dir)
Tools.print('Time (sec): {}'.format(time.time() - start)) # 356s=6min


#################################################################################
def plot_histo_graphs(dataset, title):
    graph_sizes = []
    for graph in dataset:
        graph_sizes.append(graph[0].number_of_nodes())
        # graph_sizes.append(graph[0].number_of_edges())
    plt.figure(1)
    plt.hist(graph_sizes, bins=20)
    plt.title(title)
    plt.show()
    graph_sizes = torch.Tensor(graph_sizes)
    Tools.print('nb/min/max : {} {} {}'.format(
        len(graph_sizes), graph_sizes.min().long().item(), graph_sizes.max().long().item()))
    pass


plot_histo_graphs(dataset.train, 'trainset')
plot_histo_graphs(dataset.val, 'valset')
plot_histo_graphs(dataset.test, 'testset')

Tools.print(len(dataset.train))
Tools.print(len(dataset.val))
Tools.print(len(dataset.test))
Tools.print(dataset.train[0])
Tools.print(dataset.val[0])
Tools.print(dataset.test[0])

#################################################################################
now_dataset_name = 'MNIST'
now_data_file = "D:\data\GCN\{}.pkl".format(now_dataset_name)
dataset = SuperPixDataset(now_dataset_name, data_file=now_data_file) # 54s
trainset, valset, testset = dataset.train, dataset.val, dataset.test

#################################################################################
start = time.time()
batch_size = 10
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=SuperPixDataset.collate)
Tools.print('Time (sec):',time.time() - start) # 0.0003s
