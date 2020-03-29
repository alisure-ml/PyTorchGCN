import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, ShapeNet


def data_example():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    # x: 每个节点的特征，edge_index: 图连接，edge_attr: 每条边的特征，y: 目标，pos: 位置矩阵
    data = Data(x=x, edge_index=edge_index)
    print(data)

    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data)

    print(data.keys)
    print(data['x'])
    for key, item in data:
        print("{} found in data".format(key))
        pass
    print(data.num_nodes)
    print(data.num_edges)
    print(data.num_node_features)
    print(data.num_edge_features)
    print(data.contains_isolated_nodes())
    print(data.contains_self_loops())
    print(data.is_directed())
    pass


def common_benchmark_dataset():
    dataset = 'ENZYMES'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = TUDataset(root=path, name=dataset)
    data = dataset[0]

    print(len(dataset))  # 数据集中共有600张图
    print(dataset.num_classes)
    print(dataset.num_node_features)
    print(data)  # 每张图包含37个节点，每个节点包含3个特征，共84个无向边，该图属于类别1。
    print(data.is_undirected())

    # We can use slices, long or byte tensors to split the dataset.
    train_dataset = dataset[:540]
    test_dataset = dataset[540:]
    print(train_dataset)
    print(test_dataset)

    # shuffle
    dataset = dataset.shuffle()
    print(dataset)
    # 等价于
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    print(dataset)
    pass


def common_benchmark_dataset2():
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(root=path, name=dataset)
    data = dataset[0]

    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_node_features)
    print(data.is_undirected())
    print(data.train_mask.sum().item())  # 表示针对哪个节点进行训练（140个节点）
    print(data.val_mask.sum().item())  # 表示要用于验证（例如，执行早停）的节点（500个节点）
    print(data.test_mask.sum().item())  # 表示针对哪个节点进行测试（1000个节点）
    pass


def mini_batches():
    dataset = 'ENZYMES'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = TUDataset(root=path, name=dataset, use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print(batch)  # batch: 标识节点所属的图。x: 节点的特征。edge_index: 边。y: 标签
        print(batch.num_graphs) # 该批次中图的数量

        x = scatter_mean(batch.x, batch.batch, dim=0)  # 每个图的平均节点特征
        print(x.size())
        pass

    pass


def data_transforms():
    dataset = 'ShapeNet'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = ShapeNet(root=path, categories=['Airplane'])
    print(dataset[0])
    dataset = ShapeNet(root=path, categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
    print(dataset[0])
    dataset = ShapeNet(root=dataset, categories=['Airplane'],
                       pre_transform=T.KNNGraph(k=6), transform=T.RandomTranslate(0.01))
    print(dataset[0])
    pass


class Net(torch.nn.Module):

    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        pass

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    pass


def learning_methods_on_graphs():
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(root=path, name=dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print("{} {}".format(epoch, loss.item()))
        pass

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
    pass


if __name__ == '__main__':
    learning_methods_on_graphs()
    pass
