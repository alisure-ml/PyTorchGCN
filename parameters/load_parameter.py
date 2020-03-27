from parameters.parameters import *


def GatedGCN(*args, **kwargs):
    return ParametersGatedGCN(*args, **kwargs)


def GCN(*args, **kwargs):
    return ParametersGCN(*args, **kwargs)


def GAT(*args, **kwargs):
    return ParametersGAT(*args, **kwargs)


def GraphSage(*args, **kwargs):
    return ParametersGraphSage(*args, **kwargs)


def GIN(*args, **kwargs):
    return ParametersGIN(*args, **kwargs)


def MoNet(*args, **kwargs):
    return ParametersMoNet(*args, **kwargs)


def DiffPool(*args, **kwargs):
    return ParametersDiffPool(*args, **kwargs)


def MLP(*args, **kwargs):
    return ParametersMLP(*args, **kwargs)


def MLPGated(*args, **kwargs):
    return ParametersMLPGated(*args, **kwargs)


def GNNParameter(which_model, *args, **kwargs):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'DiffPool': DiffPool,
        'MLP': MLP,
        'MLPGated': MLPGated
    }

    return models[which_model](*args, **kwargs)
