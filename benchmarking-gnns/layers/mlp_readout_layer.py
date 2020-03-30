import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
    L+1层全连接：in, in/2, ..., out
"""


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_fc_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_fc_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.fc_layers = nn.ModuleList(list_fc_layers)
        self.L = L
        pass

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = F.relu(self.fc_layers[l](y))
        y = self.fc_layers[self.L](y)
        return y

    pass
