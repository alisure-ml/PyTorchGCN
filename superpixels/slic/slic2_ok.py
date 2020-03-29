# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Button

image_name = "slic_demo.jpg"

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(image_name))

numSegments = 5000
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments=numSegments, sigma=5)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# setting buttons

bMinus = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), '-')
bPlus = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), '+')


class Action(object):
    def plus(self, event):
        global numSegments
        numSegments += 20
        segments = slic(image, n_segments=numSegments, sigma=5)
        ax.imshow(mark_boundaries(image, segments))

    def minus(self, event):
        global numSegments
        numSegments -= 20
        if numSegments <= 0:
            numSegments += 20
            return
        print('Num Segments :', numSegments)
        segments = slic(image, n_segments=numSegments, sigma=5)
        ax.imshow(mark_boundaries(image, segments))


action = Action()
bMinus.on_clicked(action.minus)
bPlus.on_clicked(action.plus)

# show the plots
plt.show()