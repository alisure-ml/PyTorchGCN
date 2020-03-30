import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import dgl

"""
    <Diffpool Fuse with GNN layers and pooling layers>
    
    DIFFPOOL:
    Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, 
    Hierarchical graph representation learning with differentiable pooling (NeurIPS 2018)
    https://arxiv.org/pdf/1806.08804.pdf
    
    ! code started from dgl diffpool examples dir
"""

from layers.graphsage_layer import GraphSageLayer   # this is GraphSageLayer
from layers.diffpool_layer import DiffPoolLayer   # this is DiffPoolBatchedGraphLayer
# from .graphsage_net import GraphSageNet   # this is GraphSage
# replace BatchedDiffPool with DenseDiffPool and BatchedGraphSAGE with DenseGraphSage
from layers.tensorized.dense_graphsage_layer import DenseGraphSage
from layers.tensorized.dense_diffpool_layer import DenseDiffPool


class DiffPoolNet(nn.Module):
    """DiffPool Fuse with GNN layers and pooling layers in sequence"""

    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.concat = net_params['cat']
        self.batch_size = net_params['batch_size']
        self.link_pred_loss = []
        self.entropy_loss = []
        
        self.embedding_h = nn.Linear(net_params.in_dim, net_params.hidden_dim)
        
        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()

        self.bn = True
        self.num_aggs = 1

        # constructing layers
        # layers before diffpool
        assert net_params.L >= 3, "n_layers too few"
        self.gc_before_pool.append(GraphSageLayer(
            net_params.hidden_dim, net_params.hidden_dim, F.relu,
            net_params.dropout, net_params.sage_aggregator, net_params.residual, self.bn))
        
        for _ in range(net_params.L - 2):
            self.gc_before_pool.append(GraphSageLayer(
                net_params.hidden_dim, net_params.hidden_dim, F.relu,
                net_params.dropout, net_params.sage_aggregator, net_params.residual, self.bn))
            pass
        
        self.gc_before_pool.append(GraphSageLayer(
            net_params.hidden_dim, net_params.embedding_dim, None,
            net_params.dropout, net_params.sage_aggregator, net_params.residual))
        
        assign_dims = [net_params.assign_dim]
        if self.concat:
            # diffpool layer receive pool_emedding_dim node feature tensor and return pool_embedding_dim node embedding
            pool_embedding_dim = net_params.hidden_dim * (net_params.L - 1) + net_params.embedding_dim
        else:
            pool_embedding_dim = net_params.embedding_dim
            pass

        self.first_diffpool_layer = DiffPoolLayer(
            pool_embedding_dim, net_params.assign_dim, net_params.hidden_dim,
            F.relu, net_params.dropout, net_params.sage_aggregator, net_params.linkpred)

        gc_after_per_pool = nn.ModuleList()
        for _ in range(net_params.L - 1):
            gc_after_per_pool.append(DenseGraphSage(net_params.hidden_dim, net_params.hidden_dim, net_params.residual))
        gc_after_per_pool.append(DenseGraphSage(net_params.hidden_dim, net_params.embedding_dim, net_params.residual))

        self.gc_after_pool = nn.ModuleList()
        self.gc_after_pool.append(gc_after_per_pool)

        net_params.assign_dim = int(net_params.assign_dim * net_params.pool_ratio)
        
        self.diffpool_layers = nn.ModuleList()
        for _ in range(net_params.num_pool - 1):
            self.diffpool_layers.append(DenseDiffPool(pool_embedding_dim, net_params.assign_dim,
                                                      net_params.hidden_dim, net_params.linkpred))
            gc_after_per_pool = nn.ModuleList()
            for _ in range(net_params.L - 1):
                gc_after_per_pool.append(DenseGraphSage(net_params.hidden_dim,
                                                        net_params.hidden_dim, net_params.residual))
                pass
            gc_after_per_pool.append(DenseGraphSage(net_params.hidden_dim,
                                                    net_params.embedding_dim, net_params.residual))
            self.gc_after_pool.append(gc_after_per_pool)
            
            assign_dims.append(net_params.assign_dim)
            net_params.assign_dim = int(net_params.assign_dim * net_params.pool_ratio)
            pass

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim *  self.num_aggs * (net_params.num_pool + 1)
        else:
            self.pred_input_dim = net_params.embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, net_params.n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
                pass
            pass

        pass

    @staticmethod
    def gcn_forward(g, h, snorm_n, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h, snorm_n)
            block_readout.append(h)
        h = gc_layers[-1](g, h, snorm_n)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    @staticmethod
    def gcn_forward_tensorized(h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g, h, e, snorm_n, snorm_e):
        self.link_pred_loss = []
        self.entropy_loss = []
        
        # node feature for assignment matrix computation is the same as the
        # original node feature
        h = self.embedding_h(h)
        h_a = h

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, snorm_n, self.gc_before_pool, self.concat)

        g.ndata['h'] = g_embedding

        readout = dgl.sum_nodes(g, 'h')
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, 'h')
            out_all.append(readout)
            pass

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = self.batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[0], self.concat)
        
        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)
            pass

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[i + 1], self.concat)
            
            readout = torch.sum(h, dim=1)
            out_all.append(readout)
            
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)
                pass
            pass
        
        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = readout
        ypred = self.pred_layer(final_readout)
        return ypred

    def batch2tensor(self, batch_adj, batch_feat, node_per_pool_graph):
        """
        transform a batched graph to batched adjacency tensor and node feature tensor
        """
        batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
        adj_list = []
        feat_list = []

        for i in range(batch_size):
            start = i * node_per_pool_graph
            end = (i + 1) * node_per_pool_graph

            # 1/sqrt(V) normalization
            snorm_n = torch.FloatTensor(node_per_pool_graph, 1).fill_(1./float(node_per_pool_graph)).sqrt().to(self.device)

            adj_list.append(batch_adj[start:end, start:end])
            feat_list.append((batch_feat[start:end, :])*snorm_n)
            pass

        adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
        feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
        adj = torch.cat(adj_list, dim=0)
        feat = torch.cat(feat_list, dim=0)

        return feat, adj
    
    def loss(self, pred, label):
        '''loss function : softmax + CE'''
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
                pass
            pass
        return loss

    pass
