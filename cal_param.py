import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(64, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))


        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))


        layers.append(nn.Conv3d(256, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))



        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))


        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


class Net2(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(64, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(128, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(256, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))



        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))


        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


class Net3(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(64, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


if __name__ == '__main__':
    """
    resnet:281548163
    vGG：59903747
    light:22127747
    """
    model = Net3().to(torch.device("cpu"))
    num = view_model_param(model)
    print(num)
    pass
