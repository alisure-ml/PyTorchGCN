import torch
from torch_geometric.nn import knn_graph
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):

    """
    propagate: 消息传播
    message: 消息
    aggregate: 聚合
    update: 更新
    
    1. Add self-loops to the adjacency matrix.
    2. Linearly transform node feature matrix.
    3. Compute normalization coefficients.
    4. Normalize node features in ϕ.
    5. Sum up neighboring node features ("add" aggregation).
    6. Return new node embeddings in γ.
    """

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        pass

    def forward(self, x, edge_index):  # [N, in_channels], [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):  # [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):  # [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out

    pass


class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(), Linear(out_channels, out_channels))
        pass

    def forward(self, x, edge_index):  # [N, in_channels]， [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)  # size=[N, N]

    def message(self, x_i, x_j):  # [E, in_channels]， [E, in_channels]
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=1))  # [E, 2 * in_channels]

    def update(self, aggr_out):  # [N, out_channels]
        return aggr_out

    pass


class DynamicEdgeConv(EdgeConv):
    """
    边缘卷积实际上是一种动态卷积，它使用特征空间中的最近邻居重新计算每一层的图。
    """

    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k
        pass

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

    pass

