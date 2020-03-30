import dgl
import torch
import torch.nn as nn
from layers.mlp_readout_layer import MLPReadout


class MLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        self.gated = net_params.gated

        self.in_feat_dropout = nn.Dropout(net_params.in_feat_dropout)  # 0.0

        feat_mlp_modules = [nn.Linear(net_params.in_dim, net_params.hidden_dim, bias=True),
                            nn.ReLU(), nn.Dropout(net_params.dropout)]  # in, 168
        for _ in range(net_params.L - 1):  # L=4
            feat_mlp_modules.append(nn.Linear(net_params.hidden_dim, net_params.hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(net_params.dropout))  # 168, 168
            pass
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = nn.Linear(net_params.hidden_dim, net_params.hidden_dim, bias=True)  # 168, 168
            pass

        self.readout_mlp = MLPReadout(net_params.hidden_dim, net_params.n_classes)  # MLP: 3
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.in_feat_dropout(nodes_feat)
        h = self.feat_mlp(h)

        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h
            graphs.ndata['h'] = h
            hg = dgl.sum_nodes(graphs, 'h')
        else:
            graphs.ndata['h'] = h
            hg = dgl.mean_nodes(graphs, 'h')
            pass

        logits = self.readout_mlp(hg)
        return logits

    pass
