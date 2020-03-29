"""
A simple linear iterative clustering (SLIC) scheme for superpixel segmentation.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq as vq
from scipy.spatial.distance import cdist
import time
import scipy.stats as stats
import scipy.ndimage as ndimage
import math

IMAGE = 'slic_demo.jpg'

SUPERPIXELS = 2000
MAXITER = 100


def getMode(x):
    return stats.mode(x, axis=None)[0]


def grad(mx, I):
    #
    r, g, b, x, y = mx
    x = int(x)
    y = int(y)

    # If they are on the edge, assume the gradient to be infinit (so that the algo drop it)
    if x == 0 or x == len(I) - 1 or y == 0 or y == len(I) - 1:
        return 2 ** 31 - 1

    #
    I1 = I[x + 1][y]
    I2 = I[x - 1][y]
    I3 = I[x][y + 1]
    I4 = I[x][y - 1]

    #
    return (np.sum((I1 - I2) ** 2) + np.sum((I3 - I4) ** 2))


def reinitM():
    global M, I, s, ishape  # ...
    numPoints = len(M)
    # For each point in the cluster
    for p in range(numPoints):
        # Get the points and values
        r, g, b, x, y = M[p]
        x = int(x)
        y = int(y)
        # Compare in the encompassing 9-pixel region to find lowest gradient
        minGrad = math.inf
        min_x = 0
        min_y = 0

        for row in range(x - 1, x + 2):
            for col in range(y - 1, y + 2):
                mx = np.concatenate((I[row, col, :], [row, col]))
                mx_grad = grad(mx, I)
                if minGrad > mx_grad:
                    minGrad = mx_grad
                    min_x = row
                    min_y = col

        # Replace current vector with the vector of lowest gradient
        M[p] = np.concatenate((I[min_x, min_y, :], [min_x, min_y]))

        # pass


def convergedYet():
    global M_prev, M, iteration
    thres = 7
    error = 0

    for i in range(len(M)):
        error_vector = [M[i][0] - M_prev[i][0], M[i][1] - M_prev[i][1], M[i][2] - M_prev[i][2]]
        error = np.sqrt(error_vector[0] ** 2 + error_vector[1] ** 2 + error_vector[2] ** 2)
        print(type(error))
    if error < thres:
        return True
    else:
        print("Percent done ", (thres / error) * 100)
        M_prev = M.copy()
        return False


I = plt.imread(IMAGE).astype('float')
ishape = I.shape

# s = the sampling interval.
s = int(np.round(np.sqrt((ishape[0] * ishape[1]) // SUPERPIXELS)))

# cs = the color distance scaler square, see Eq. 10-91
cs = 3 * 255 ** 2  # Just the maximum color discrepancy: weights regular regions highly.
# cs = 100**2 # Trying for image boundaries instead of spatial regularity.

# SLIC Step 1.
# Make M, the initial cluster centers. This should be SUPERPIXELS x 5, for
# the spatial coords of the cluster centers and the average color of them.
M = np.concatenate([np.expand_dims(IC, axis=1) for IC in [I[s::s, s::s, 0].ravel(),
                                                          I[s::s, s::s, 1].ravel(), I[s::s, s::s, 2].ravel()]], axis=1)
xm = np.meshgrid(np.arange(s, ishape[0], s), np.arange(s, ishape[1], s), indexing='ij')
M = np.append(M, np.concatenate([np.expand_dims(xi, axis=1) for xi in [x.ravel() for x in xm]], axis=1), axis=1)

# Here we should reinitialize these M to the lowest-gradient point in the 3x3,
# but we won't yet.
reinitM()

# Initial distance measure D for every pixel, and label
D = 1.0e20 * np.ones((ishape[0], ishape[1]))
L = -1 * np.ones((ishape[0], ishape[1]))

# xs are the x and y coords of every pixel. used for average computing later.
xs = np.meshgrid(np.arange(ishape[0]), np.arange(ishape[1]), indexing='ij')

# Stores the initial cluster centers
M_prev = np.zeros(M.shape)

iteration = 15

while True:
    # SLIC Step 2
    # First, loop over clusters and assign pixels to them.
    for mi, mx in enumerate(M):
        # mi is cluster number, mx is 1x5 of the average color and position of the cluster.
        i, j = [int(q) for q in np.round(mx[3:5])]
        # look over all pixels in the +- 2s neighborhood to see if they belong
        # to this cluster. See DIP 10.5

        for xx in range(i - 2 * s, i + 2 * s + 1):
            for yy in range(j - 2 * s, j + 2 * s + 1):
                if xx >= 0 and yy >= 0 and xx < len(D) and yy < len(D[0]):

                    rd, gd, bd = I[xx, yy]  # Color components of the pixel xx, yy

                    # Compute positional distance
                    ds = np.sqrt((xx - i) ** 2 + (yy - j) ** 2)

                    # Computer color distance
                    dc = np.sqrt((rd - mx[0]) ** 2 + (gd - mx[1]) ** 2 + (bd - mx[2]) ** 2)

                    #
                    # Assign a value to each component of D by calculating pixel to cluster distance (positional and color distance)
                    #
                    d = np.sqrt((dc) ** 2 + (ds) ** 2 / s * cs)

                    # Notation differs slightly from the SLIC paper
                    if D[xx][yy] > d:
                        D[xx][yy] = d
                        L[xx][yy] = mi

        print('looped over mi %d' % mi)

    M_prev = M.copy()
    # SLIC Step 3
    # Then, recompute clusters.
    for mi in range(len(M)):
        whichAreMi = (L == mi)
        if np.any(whichAreMi):  # only if there are any pixels labeled mi
            M[mi, :3] = np.mean(I[whichAreMi, :], axis=0)
            M[mi, 3] = np.mean(xs[0][whichAreMi])
            M[mi, 4] = np.mean(xs[1][whichAreMi])

    # SLIC Step 4
    # Then, test for convergence.
    iteration += 1
    if convergedYet():  # error convergence test, not implemented yet.
        break

# SLIC Step 5
# After the clustering, clean up the label image:
ndimage.generic_filter(L, function=getMode, size=5, output=L, mode='reflect')

# Reconstruct the image, where every pixel is replaced with its superpixel's average color.
# IR = np.zeros(I.shape)
# for mi in range(len(M)):
#     IR[L == mi, :] = M[mi,:3]
# The above is more complicated than it needs to be. This crazy line does it all.
IR = M[L.astype(int), :3]

f, ax = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
ax[0].imshow(I / 255)
ax[0].set_title('Original Image')
ax[1].imshow(IR / 255)
ax[1].set_title('%d Superpixels after %d Iter' % (SUPERPIXELS, iteration))
plt.show()
