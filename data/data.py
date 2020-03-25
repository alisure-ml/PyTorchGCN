from data.superpixels import SuperPixDataset


def LoadData(DATASET_NAME):
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    pass
