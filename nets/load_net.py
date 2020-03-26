from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.gin_net import GINNet
from nets.mo_net import MoNet as MoNet_
from nets.diffpool_net import DiffPoolNet
from nets.mlp_net import MLPNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def MoNet(net_params):
    return MoNet_(net_params)


def DiffPool(net_params):
    return DiffPoolNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)


def gnn_model(params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'DiffPool': DiffPool,
        'MLP': MLP
    }

    return models[params.model_name](params)
