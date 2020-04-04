import os
import dgl
import time
import torch
import pickle
import numpy as np
import torch.utils.data
from alisuretool.Tools import Tools
from scipy.spatial.distance import cdist


# 通过坐标和位置计算邻接矩阵：N*N
def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):

    def sigma(dists, kth=8):
        # Compute sigma and reshape
        try:
            # Get k-nearest neighbors for each node
            knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
            sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth
        except ValueError:  # handling for graphs with num_nodes less than kth
            num_nodes = dists.shape[0]
            # this sigma value is irrelevant since not used for final compute_edge_list
            sigma = np.array([1] * num_nodes).reshape(num_nodes, 1)

        return sigma + 1e-8  # adding epsilon to avoid zero value of sigma

    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)

    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist / sigma(c_dist, kth)) ** 2 - (f_dist / sigma(f_dist, kth)) ** 2)
    else:
        A = np.exp(- (c_dist / sigma(c_dist, kth)) ** 2)

    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


# 通过邻接矩阵计算K近邻以及权重：N*N
def compute_edges_list(A, kth=8 + 1):
    # Get k-similar neighbor indices for each node
    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > 9:
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]  # NEW
    else:
        # handling for graphs with less than kth nodes. In such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A  # NEW

        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)  # NEW
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)
    return knns, knn_values  # NEW


class SuperPixDGL(torch.utils.data.Dataset):

    def __init__(self, data_dir, dataset, split, use_mean_px=True, use_coord=True):
        self.split = split
        self.graph_lists = []

        if dataset == 'MNIST':
            self.img_size = 28
            with open(os.path.join(data_dir, 'mnist_75sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
                pass
        elif dataset == 'CIFAR10':
            self.img_size = 32
            with open(os.path.join(data_dir, 'cifar10_150sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
                pass
            pass

        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        self.n_samples = len(self.labels)

        self._prepare()
        pass

    def _prepare(self):
        Tools.print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size

            A = compute_adjacency_matrix_images(coord, mean_px, self.use_mean_px)  # super-pixel locations + features
            edges_list, edge_values_list = compute_edges_list(A)  # NEW
            edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !

            x = np.concatenate((mean_px.reshape(A.shape[0], -1), coord.reshape(A.shape[0], 2)), axis=1)

            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
            self.edge_features.append(edge_values_list)
            self.node_features.append(x)
            pass

        for index in range(len(self.sp_data)):
            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index]).half()

            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts != src])
                pass

            # adding edge features for Residual Gated ConvNet
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW 

            self.graph_lists.append(g)
            pass

        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

    pass


class SuperPixDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name, num_val=5000, use_mean_px=True, use_coord=True, data_dir=None):
        t_data = time.time()
        self.name = name

        if use_mean_px:
            Tools.print('Adj matrix defined from super-pixel locations + features')
        else:
            Tools.print('Adj matrix defined from super-pixel locations (only)')

        data_dir = data_dir if data_dir else "data/superpixels"
        self.train_ = SuperPixDGL(data_dir, self.name, split='train', use_mean_px=use_mean_px, use_coord=use_coord)
        self.test = SuperPixDGL(data_dir, self.name,  split='test', use_mean_px=use_mean_px, use_coord=use_coord)

        _train_graphs, _train_labels = self.train_[num_val:]
        _val_graphs, _val_labels = self.train_[:num_val]

        self.train = DGLFormDataset(_train_graphs, _train_labels)
        self.val = DGLFormDataset(_val_graphs, _val_labels)

        Tools.print("[I] Data load time: {:.4f}s".format(time.time() - t_data))
        pass

    pass


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]
        pass

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

    pass


class SuperPixDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_file=None):
        start = time.time()
        Tools.print("[I] Loading dataset {}".format(name))
        self.name = name
        data_file = data_file if data_file else "data/superpixels/{}.pkl".format(name)
        with open(data_file, "rb") as f:
            f = pickle.load(f)
            self.train, self.val, self.test = f[0], f[1], f[2]
            pass

        Tools.print("train, test, val sizes : {} {} {}".format(len(self.train), len(self.test), len(self.val)))
        Tools.print("[I] Finished loading.")
        Tools.print("[I] Data load time: {:.4f}s".format(time.time() - start))
        pass

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))

        nodes_num = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        edges_num = [graphs[i].number_of_edges() for i in range(len(graphs))]
        nodes_num_norm = [torch.FloatTensor(num, 1).fill_(1. / float(num)) for num in nodes_num]
        edges_num_norm = [torch.FloatTensor(num, 1).fill_(1. / float(num)) for num in edges_num]
        nodes_num_norm_sqrt = torch.cat(nodes_num_norm).sqrt()
        edges_num_norm_sqrt = torch.cat(edges_num_norm).sqrt()

        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
            pass

        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, nodes_num_norm_sqrt, edges_num_norm_sqrt

    def add_self_loops(self):
        # function for adding self loops
        self.train.graph_lists = [self.self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self.self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self.self_loop(g) for g in self.test.graph_lists]

        self.train = DGLFormDataset(self.train.graph_lists, self.train.graph_labels)
        self.val = DGLFormDataset(self.val.graph_lists, self.val.graph_labels)
        self.test = DGLFormDataset(self.test.graph_lists, self.test.graph_labels)
        pass

    @staticmethod
    def self_loop(g):
        """
            Utility function only, to be used only when necessary as per user self_loop flag
            : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
            This function is called inside a function in SuperPixDataset class.
        """
        new_g = dgl.DGLGraph()
        new_g.add_nodes(g.number_of_nodes())
        new_g.ndata['feat'] = g.ndata['feat']

        src, dst = g.all_edges(order="eid")
        src = dgl.backend.zerocopy_to_numpy(src)
        dst = dgl.backend.zerocopy_to_numpy(dst)
        non_self_edges_idx = src != dst
        nodes = np.arange(g.number_of_nodes())
        new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
        new_g.add_edges(nodes, nodes)

        # This new edata is not used since this function gets called only for GCN, GAT
        # However, we need this for the generic requirement of ndata and edata
        new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
        return new_g

    pass
