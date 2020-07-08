## 视觉嵌入深度图卷积


### 超像素

> 先采用`SLIC`方法。希望该步骤在0.01秒内完成。


### 视觉嵌入

> 采用深度自编码网络对每一个超像素进行视觉嵌入，包括形状和纹理

* 嵌入流程

    图像 =》超像素 =》嵌入 =》重构

* 嵌入输出

    1. 节点特征：每个超像素输出一个一维的视觉嵌入特征+超像素的位置和大小

    2. 边特正：邻接矩阵或K近邻矩阵


### 图卷积

* 网络输入

    视觉嵌入的输出

* 网络

    1. GCN
    2. GAT
    3. GraphSage
    4. GIN
    5. MoNet
    6. GatedGCN

* 网络输出

    分类结果


### 数据集

* MNIST

* CIFAR10

* ImageNet


### 结果


### PyG

```
pip install torch-scatter==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

pip uninstall torch-scatter
pip uninstall torch-sparse
pip uninstall torch-cluster
pip uninstall torch-spline-conv
pip uninstall torch-geometric

pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

```
LD_LIBRARY_PATH=/usr/local/cuda/lib64
```
