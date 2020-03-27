import dgl
import torch.nn as nn
import torch.nn.functional as F


"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout


class GCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        self.net_params = net_params
        self.readout = self.net_params.readout

        self.embedding_h = nn.Linear(self.net_params.in_dim, self.net_params.hidden_dim)
        self.in_feat_dropout = nn.Dropout(self.net_params.in_feat_dropout)

        self.layers = nn.ModuleList([GCNLayer(
            self.net_params.hidden_dim, self.net_params.hidden_dim, F.relu,
            self.net_params.dropout, self.net_params.graph_norm, self.net_params.batch_norm,
            self.net_params.residual) for _ in range(self.net_params.L - 1)])

        self.layers.append(GCNLayer(
            self.net_params.hidden_dim, self.net_params.out_dim, F.relu, self.net_params.dropout,
            self.net_params.graph_norm, self.net_params.batch_norm, self.net_params.residual))

        self.readout_mlp = MLPReadout(self.net_params.out_dim, self.net_params.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        hidden_nodes_feat = self.in_feat_dropout(hidden_nodes_feat)
        for conv in self.layers:
            hidden_nodes_feat = conv(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        graphs.ndata['h'] = hidden_nodes_feat
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
