import os
import time
from skimage import io
import matplotlib.pyplot as plt
from skimage import segmentation
from alisuretool.Tools import Tools


def demo(image_name="input\\3.jpg", n_segment=1024):
    image = io.imread(image_name)

    start = time.time()
    Tools.print("start")
    segment = segmentation.slic(image, n_segments=n_segment, sigma=5)
    Tools.print("end {} {}".format(n_segment, time.time() - start))

    result = segmentation.mark_boundaries(image, segment)
    fig = plt.figure("{}".format(n_segment))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(result)
    plt.axis("off")
    plt.show()

    pass


def demo_multi(image_name="input\\3.jpg", n_segments=[2 ** (i + 1) for i in range(12)]):
    image = io.imread(image_name)

    start = time.time()
    Tools.print("start")
    segments = {}
    for n_segment in n_segments:
        start_one = time.time()
        segment = segmentation.slic(image, n_segments=n_segment, sigma=2)
        segments[n_segment] = segment
        Tools.print("end {} {}".format(n_segment, time.time() - start_one))
        pass
    Tools.print("end {}".format(time.time() - start))

    show_n = n_segments[-1]
    result = segmentation.mark_boundaries(image, segments[show_n], mode="thick")
    fig = plt.figure("{}".format(show_n))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(result)
    plt.axis("off")
    plt.show()
    pass


if __name__ == '__main__':
    demo_multi(image_name="input\\11.jpg", n_segments=[2 ** (i + 1) for i in range(8)])
    pass
