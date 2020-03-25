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
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.graph_norm,
                                              self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout,
                                    self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        hidden_nodes_feat = self.in_feat_dropout(hidden_nodes_feat)
        for conv in self.layers:
            hidden_nodes_feat = conv(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
        graphs.ndata['h'] = hidden_nodes_feat

        if self.readout == "sum":
            global_feat = dgl.sum_nodes(graphs, 'h')
        elif self.readout == "max":
            global_feat = dgl.max_nodes(graphs, 'h')
        elif self.readout == "mean":
            global_feat = dgl.mean_nodes(graphs, 'h')
        else:
            global_feat = dgl.mean_nodes(graphs, 'h')  # default readout is mean nodes
            pass

        return self.MLP_layer(global_feat)

    def loss(self, pred, label):
        loss = nn.CrossEntropyLoss()(pred, label)
        return loss

    pass
