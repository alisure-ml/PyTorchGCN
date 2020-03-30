import torch
import torch.nn as nn
from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""


class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        self.embedding_h = nn.Linear(net_params.in_dim, net_params.hidden_dim)

        self.ginlayers = torch.nn.ModuleList()
        for layer in range(net_params.L):
            mlp = MLP(net_params.n_mlp_GIN, net_params.hidden_dim, net_params.hidden_dim, net_params.hidden_dim)
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), net_params.neighbor_aggr_GIN, net_params.dropout,
                                           net_params.graph_norm, net_params.batch_norm,
                                           net_params.residual, 0, net_params.learn_eps_GIN))
            pass

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(net_params.hidden_dim, net_params.n_classes))
            pass
        
        if net_params.readout == 'sum':
            self.pool = SumPooling()
        elif net_params.readout == 'mean':
            self.pool = AvgPooling()
        elif net_params.readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        pass
        
    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](graphs, h, nodes_num_norm_sqrt)
            hidden_rep.append(h)
            pass

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(graphs, h)
            score_over_layer += self.linears_prediction[i](pooled_h)
            pass

        return score_over_layer
        
    pass
