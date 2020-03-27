import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout


class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()
        self.readout = net_params.readout

        self.embedding_h = nn.Linear(net_params.in_dim, net_params.hidden_dim)
        self.in_feat_dropout = nn.Dropout(net_params.in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(
            net_params.hidden_dim, net_params.hidden_dim, F.relu, net_params.dropout,
            net_params.sage_aggregator, net_params.residual) for _ in range(net_params.L - 1)])
        self.layers.append(GraphSageLayer(net_params.hidden_dim, net_params.out_dim, F.relu,
                                          net_params.dropout, net_params.sage_aggregator, net_params.residual))
        self.readout_mlp = MLPReadout(net_params.out_dim, net_params.n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(graphs, h, nodes_num_norm_sqrt)
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