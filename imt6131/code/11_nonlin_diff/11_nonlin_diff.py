#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('lena.png')
im = np.sum(im, 2) / 3.

dt = .25
eps = 1e-5
kappa = 1e4

gx, gy = np.gradient(im)
gnormsq = gx**2 + gy**2
D = 1 / (1 + kappa * gnormsq)

for i in range(100):
    plt.imsave('im_%04d.png' % i, im, cmap=plt.cm.gray, vmin=0, vmax=1)
    gx, gy = np.gradient(im)
    gxx, tmp = np.gradient(D * gx)
    tmp, gyy = np.gradient(D * gy)
    tv = gxx + gyy
    im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
