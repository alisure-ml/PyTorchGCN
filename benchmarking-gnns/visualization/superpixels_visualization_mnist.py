import os
import dgl
import time
import torch
import random
import pickle
import matplotlib
import numpy as np
import networkx as nx
from scipy import ndimage
from pylab import rcParams
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools
from scipy.spatial.distance import cdist
from torchvision import transforms, datasets
from scipy.spatial.distance import pdist, squareform


def sigma(dists, kth=8):
    knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
    sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth
    return sigma + 1e-8


def compute_adjacency_matrix_images(coord, feat, use_feat=False, kth=8):
    coord = coord.reshape(-1, 2)
    c_dist = cdist(coord, coord)  # Compute coordinate distance

    if use_feat:
        f_dist = cdist(feat, feat)  # Compute feature distance
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2 - (f_dist / sigma(f_dist)) ** 2)  # Compute adjacency
    else:
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2)

    A = 0.5 * (A + A.T)  # Convert to symmetric matrix
    A[np.diag_indices_from(A)] = 0  # A = 0.5 * A * A.T
    return A


def compute_edges_list(A, kth=8 + 1):
    # Get k-similar neighbor indices for each node
    if 1 == 1:
        num_nodes = A.shape[0]
        new_kth = num_nodes - kth
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knns_d = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
    else:
        knns = np.argpartition(A, kth, axis=-1)[:, kth::-1]
        knns_d = np.partition(A, kth, axis=-1)[:, kth::-1]
    return knns, knns_d


class MNISTSuperPix(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, use_mean_px=True, use_coord=True, use_feat_for_graph_construct=False):
        self.split = split
        self.is_test = split.lower() in ['test', 'val']
        with open(os.path.join(data_dir, 'mnist_75sp_%s.pkl' % split), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)
            pass

        self.use_mean_px = use_mean_px
        self.use_feat_for_graph = use_feat_for_graph_construct
        self.use_coord = use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28
        pass

    def precompute_graph_images(self):
        Tools.print('precompute all data for the %s set...' % self.split.upper())
        self.Adj_matrices, self.node_features, self.edges_lists = [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord, mean_px, use_feat=self.use_feat_for_graph)
            edges_list, _ = compute_edges_list(A)
            N_nodes = A.shape[0]

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                x = np.concatenate((x, coord), axis=1) if self.use_mean_px else coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features

            self.node_features.append(x)
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
            pass
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        g = dgl.DGLGraph()
        g.add_nodes(self.node_features[index].shape[0])
        g.ndata['feat'] = torch.Tensor(self.node_features[index])
        for src, dsts in enumerate(self.edges_lists[index]):
            g.add_edges(src, dsts[dsts != src])

        return g, self.labels[index]

    pass


#################################################################################
# Taking the test dataset only for sample visualization
tt = time.time()
now_data_dir = "D:\data\GCN\superpixels"
data_no_feat_knn = MNISTSuperPix(now_data_dir, split='train', use_feat_for_graph_construct=False)
data_no_feat_knn.precompute_graph_images()
Tools.print("Time taken: {:.4f}s".format(time.time()-tt))

#################################################################################
tt = time.time()
data_with_feat_knn = MNISTSuperPix(now_data_dir, split='train', use_feat_for_graph_construct=True)
data_with_feat_knn.precompute_graph_images()
Tools.print("Time taken: {:.4f}s".format(time.time()-tt))

#################################################################################
MNIST_IMAGE_DIR = 'D:\data\MNIST'
dataset = datasets.MNIST(root=MNIST_IMAGE_DIR, train=True, download=True, transform=transforms.ToTensor())
x, _ = dataset[777] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
sample = np.random.choice(len(data_no_feat_knn))
g_sample = data_no_feat_knn[sample][0]
Tools.print("Label: ".format(data_no_feat_knn[sample][1]))
nx.draw(g_sample.to_networkx(), with_labels=True)
plt.show()


#################################################################################
def plot_superpixels_graph(plt, sp_data, adj_matrix, label, feat_coord, with_edges):
    Y = squareform(pdist(sp_data[1], 'euclidean'))
    x_coord = sp_data[1]  # np.flip(dataset.train.sp_data[_][1], 1)
    intensities = sp_data[0].reshape(-1)

    G = nx.from_numpy_matrix(Y)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    rotated_pos = {node: (y, -x) for (node, (x, y)) in pos.items()}  # rotate the coords by 90 degree

    edge_list = []
    for src, dsts in enumerate(compute_edges_list(adj_matrix)[0]):
        for dst in dsts:
            edge_list.append((src, dst))
            pass
        pass

    nx.draw_networkx_nodes(G, rotated_pos, node_color=intensities, cmap=matplotlib.cm.Reds, node_size=60)
    if with_edges:
        nx.draw_networkx_edges(G, rotated_pos, edge_list, alpha=0.3)
    title = "Label: " + str(label)
    title += " | Using feat and coord for knn" if feat_coord else " | Using only coord for knn"
    if not with_edges:
        title = "Label: " + str(label) + " | Only superpixel nodes"

    plt.title.set_text(title)
    pass


def show_image(plt, idx, alpha):
    x, label = dataset[idx]  # x is now a torch.Tensor
    img = x.numpy()[0]

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title.set_text("Label: " + str(label) + " | Original Image")
    pass


num_samples_plot = 3
for f_idx, idx in enumerate(np.random.choice(int(len(data_no_feat_knn)/2), num_samples_plot, replace=False)):
    f = plt.figure(f_idx, figsize=(23, 5))
    plt1 = f.add_subplot(141)
    show_image(plt1, idx, alpha=0.5)

    plt2 = f.add_subplot(142)
    plot_superpixels_graph(plt2, data_no_feat_knn.sp_data[idx], data_no_feat_knn.Adj_matrices[idx],
                           data_no_feat_knn[idx][1], data_no_feat_knn.use_feat_for_graph, with_edges=False)

    plt3 = f.add_subplot(143)
    plot_superpixels_graph(plt3, data_no_feat_knn.sp_data[idx], data_no_feat_knn.Adj_matrices[idx],
                           data_no_feat_knn[idx][1], data_no_feat_knn.use_feat_for_graph, with_edges=True)

    plt4 = f.add_subplot(144)
    plot_superpixels_graph(plt4, data_with_feat_knn.sp_data[idx], data_with_feat_knn.Adj_matrices[idx],
                           data_with_feat_knn[idx][1], data_with_feat_knn.use_feat_for_graph, with_edges=True)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    f.savefig('mnist_superpix_'+str(idx)+'.jpg')
    plt.show()
    pass


Tools.print(compute_edges_list(data_no_feat_knn.Adj_matrices[0])[1][:10])
Tools.print(compute_edges_list(data_with_feat_knn.Adj_matrices[0])[1][:10])
