import torch
import torch.nn as nn

import dgl

import numpy as np

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout


class MoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        self.readout = net_params.readout
        self.device = net_params.device
        self.aggr_type = "sum"  # default for MoNet

        self.embedding_h = nn.Linear(net_params.in_dim, net_params.hidden_dim)

        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(net_params.L - 1):
            self.layers.append(GMMLayer(net_params.hidden_dim, net_params.hidden_dim, net_params.pseudo_dim_MoNet,
                                        net_params.kernel, self.aggr_type, net_params.dropout, net_params.graph_norm,
                                        net_params.batch_norm, net_params.residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, net_params.pseudo_dim_MoNet), nn.Tanh()))
            pass

        # Output layer
        self.layers.append(GMMLayer(net_params.hidden_dim, net_params.out_dim, net_params.pseudo_dim_MoNet,
                                    net_params.kernel, self.aggr_type, net_params.dropout, net_params.graph_norm,
                                    net_params.batch_norm, net_params.residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, net_params.pseudo_dim_MoNet), nn.Tanh()))

        self.readout_mlp = MLPReadout(net_params.out_dim, net_params.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)

        # computing the 'pseudo' named tensor which depends on node degrees
        us, vs = graphs.edges()
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        pseudo = [[1 / np.sqrt(graphs.in_degree(us[i]) + 1), 1 / np.sqrt(
            graphs.in_degree(vs[i]) + 1)] for i in range(graphs.number_of_edges())]
        pseudo = torch.Tensor(pseudo).to(self.device)

        for i in range(len(self.layers)):
            h = self.layers[i](graphs, h, self.pseudo_proj[i](pseudo), nodes_num_norm_sqrt)
        graphs.ndata['h'] = h
        hg = self.readout_fn(self.readout, graphs, 'h')
        logits = self.readout_mlp(hg)

        return logits

    @staticmethod
    def readout_fn(readout, graphs, h):
        if readout == "sum":
            hg = dgl.sum_nodes(graphs, h)
        elif readout == "max":
            hg = dgl.max_nodes(graphs, h)
        elif readout == "mean":
            hg = dgl.mean_nodes(graphs, h)
        else:
            hg = dgl.mean_nodes(graphs, h)  # default readout is mean nodes
        return hg

    pass
