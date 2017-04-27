#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('lena.png')
im = np.sum(im, 2) / 3.
im = im + .05 * np.random.randn(np.shape(im)[0], np.shape(im)[1])
im[im > 1] = 1
im[im < 0] = 0
im0 = im.copy()

dt = .2
lambd = .1

for i in range(100):
    plt.imsave('im_%04d.png' % i, im, cmap=plt.cm.gray, vmin=0, vmax=1)
    gradx = im[:,1:] - im[:,:-1]
    grady = im[1:,:] - im[:-1,:]
    laplace = gradx[1:-1,1:] - gradx[1:-1,:-1] + \
              grady[1:,1:-1] - grady[:-1,1:-1]
    im[1:-1,1:-1] = im[1:-1,1:-1] + dt * laplace - \
      dt * lambd * (im[1:-1,1:-1] - im0[1:-1,1:-1])
