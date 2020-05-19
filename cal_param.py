import numpy as np
import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(10, 20, kernel_size=3, stride=1, padding=1, bias=bias))
        layers.append(nn.Conv3d(30, 30, kernel_size=3, stride=1, padding=1, bias=bias))
        layers.append(nn.Conv3d(30, 30, kernel_size=3, stride=1, padding=1, bias=bias))
        layers.append(nn.Conv3d(30, 30, kernel_size=3, stride=1, padding=1, bias=bias))
        layers.append(nn.Conv3d(30, 30, kernel_size=3, stride=1, padding=1, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    def forward(self, x):
        e = self.features(x)
        return e

    pass


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


if __name__ == '__main__':
    model = Net().to(torch.device("cpu"))
    num = view_model_param(model)
    print(num)
    pass
