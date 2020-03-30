"""
    视觉嵌入：图像 =》超像素 =》嵌入 =》重构
    输出为图卷积的输入
"""
import os
import cv2
import time
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import segmentation
from alisuretool.Tools import Tools


class DealSuperPixel(object):

    def __init__(self, image_data, ds_image_size=224, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        self.ds_image_size = ds_image_size
        self.super_pixel_size = super_pixel_size
        self.super_pixel_num = (self.ds_image_size // self.super_pixel_size) ** 2

        self.image_data = image_data if len(image_data) == self.ds_image_size else cv2.resize(
            image_data, (self.ds_image_size, self.ds_image_size))

        self.slic_sigma = slic_sigma
        self.slic_max_iter = slic_max_iter
        pass

    def run(self):
        start = time.time()
        segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                    sigma=self.slic_sigma, max_iter=self.slic_max_iter)
        end = time.time() - start
        Tools.print("SLIC time {} {}".format(self.super_pixel_num, end))

        start = time.time()
        super_pixel_info = {}
        for i in range(segment.max() + 1):
            _now_i = segment == i
            _now_where = np.argwhere(_now_i)
            x_min, x_max = _now_where[:, 0].min(), _now_where[:, 0].max()
            y_min, y_max = _now_where[:, 1].min(), _now_where[:, 1].max()

            super_pixel_size = len(_now_where)  # 大小
            assert super_pixel_size > 0
            super_pixel_area = (x_min, x_max, y_min, y_max)  # 坐标
            super_pixel_label = np.asarray(_now_i[x_min: x_max + 1, y_min: y_max + 1], dtype=np.int)  # 是否属于超像素

            _super_pixel_label_3 = np.expand_dims(super_pixel_label, axis=-1)
            _super_pixel_label_3 = np.concatenate([_super_pixel_label_3,
                                                   _super_pixel_label_3, _super_pixel_label_3], axis=-1)

            super_pixel_data = self.image_data[x_min: x_max + 1, y_min: y_max + 1]  # 属于超像素所在矩形区域的值
            super_pixel_data2 = super_pixel_data * _super_pixel_label_3  # 属于超像素的值

            # 计算邻接矩阵
            _x_min_a = x_min - (1 if x_min > 0 else 0)
            _y_min_a = y_min - (1 if y_min > 0 else 0)
            _x_max_a = x_max + 1 + (1 if x_max < len(segment) else 0)
            _y_max_a = y_max + 1 + (1 if y_max < len(segment[0]) else 0)
            super_pixel_area_large = segment[_x_min_a: _x_max_a, _y_min_a: _y_max_a]
            super_pixel_unique_id = np.unique(super_pixel_area_large)
            super_pixel_adjacency = [sp_id for sp_id in super_pixel_unique_id if sp_id != i]

            super_pixel_info[i] = {"size": super_pixel_size, "area": super_pixel_area,
                                   "label": super_pixel_label, "data": super_pixel_data,
                                   "data2": super_pixel_data2, "adj": super_pixel_adjacency}
            pass

        adjacency_info = []
        for super_pixel_id in super_pixel_info:
            now_adj = super_pixel_info[super_pixel_id]["adj"]
            now_area = super_pixel_info[super_pixel_id]["area"]

            _adjacency_area = [super_pixel_info[sp_id]["area"] for sp_id in now_adj]
            _now_center = ((now_area[0] + now_area[1]) / 2, (now_area[2] + now_area[3]) / 2)
            _adjacency_center = [((area[0] + area[1]) / 2, (area[2] + area[3]) / 2) for area in _adjacency_area]

            adjacency_dis = [np.sqrt((_now_center[0] - center[0]) ** 2 +
                                     (_now_center[1] - center[1]) ** 2) for center in _adjacency_center]
            softmax_w = self._softmax_of_distance(adjacency_dis)
            adjacency_w = [(super_pixel_id, adj_id, softmax_w[ind]) for ind, adj_id in enumerate(now_adj)]

            adjacency_info.extend(adjacency_w)
            pass

        end = time.time() - start
        Tools.print("SLIC time {} {}".format(self.super_pixel_num, end))
        return segment, super_pixel_info, adjacency_info

    def show(self, segment):
        result = segmentation.mark_boundaries(self.image_data, segment)
        fig = plt.figure("{}".format(self.super_pixel_num))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(result)
        plt.axis("off")
        plt.show()
        pass

    @staticmethod
    def _softmax_of_distance(distance):
        distance = np.sum(np.asarray(distance)) / np.asarray(distance)
        return np.exp(distance) / np.sum(np.exp(distance), axis=0)

    pass

if __name__ == '__main__':
    now_image_name = "data\\input\\1.jpg"
    now_image_data = io.imread(now_image_name)
    deal_super_pixel = DealSuperPixel(image_data=now_image_data, ds_image_size=224)
    now_segment, now_super_pixel_info, now_adjacency_info = deal_super_pixel.run()
    deal_super_pixel.show(now_segment)
    pass
