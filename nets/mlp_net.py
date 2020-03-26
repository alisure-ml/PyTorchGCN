import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout


class MLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        self.gated = net_params.gated

        self.in_feat_dropout = nn.Dropout(net_params.in_feat_dropout)

        feat_mlp_modules = [nn.Linear(net_params.in_dim, net_params.hidden_dim, bias=True),
                            nn.ReLU(), nn.Dropout(net_params.dropout)]
        for _ in range(net_params.L - 1):
            feat_mlp_modules.append(nn.Linear(net_params.hidden_dim, net_params.hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(net_params.dropout))
            pass

        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = nn.Linear(net_params.hidden_dim, net_params.hidden_dim, bias=True)
            pass

        self.readout_mlp = MLPReadout(net_params.hidden_dim, net_params.n_classes)
        pass

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.in_feat_dropout(h)
        h = self.feat_mlp(h)
        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h
            g.ndata['h'] = h
            hg = dgl.sum_nodes(g, 'h')
        else:
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            pass

        return self.readout_mlp(hg)

    def loss(self, pred, label):
        return nn.CrossEntropyLoss()(pred, label)

    pass
