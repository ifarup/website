#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('lena.png')
im = np.sum(im, 2) / 3.

dt = .01
eps = 1e-5

for i in range(100):
    plt.imsave('im_%04d.png' % i, im, cmap=plt.cm.gray, vmin=0, vmax=1)
    gx, gy = np.gradient(im)
    gradnorm = np.sqrt(gx**2 + gy**2) + eps
    gxx,tmp = np.gradient(gx / gradnorm)
    tmp, gyy = np.gradient(gy / gradnorm)
    tv = gxx + gyy
    im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
