import torch
import torch.nn as nn

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout


class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        self.readout = net_params['readout']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(net_params.in_dim, net_params.hidden_dim)
        self.embedding_e = nn.Linear(net_params.in_dim_edge, net_params.hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayer(net_params.hidden_dim, net_params.hidden_dim,
                                                    net_params.dropout, net_params.graph_norm, net_params.batch_norm,
                                                    net_params.residual) for _ in range(net_params.L-1)])
        self.layers.append(GatedGCNLayer(net_params.hidden_dim, net_params.out_dim, net_params.dropout,
                                         net_params.graph_norm, net_params.batch_norm, net_params.residual))
        self.readout_mlp = MLPReadout(net_params.out_dim, net_params.n_classes)
        pass
        
    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        edges_feat = edges_feat if self.edge_feat else torch.ones_like(edges_feat).to(self.device)

        # input embedding
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)
        
        # convnets
        for conv in self.layers:
            h, e = conv(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

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
