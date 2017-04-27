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
gnorm = np.sqrt(gnormsq)

v1x = np.zeros(np.shape(gx))
v1y = np.zeros(np.shape(gx))
v2x = np.zeros(np.shape(gx))
v2y = np.zeros(np.shape(gx))
lambda1 = np.zeros(np.shape(gx))
lambda2 = np.zeros(np.shape(gx))

v1x[...] = 1
v1y[...] = 0
v2x[...] = 0
v2y[...] = 1
lambda1[...] = 1
lambda2[...] = 1

ind = (gnormsq != 0)
v1x[ind] = gx[ind] / gnorm[ind]
v1y[ind] = gy[ind] / gnorm[ind]
v2x[ind] = v1y[ind]
v2y[ind] = -v1x[ind]
lambda1[ind] = 1 / (1 + kappa * gnormsq[ind])
lambda2[ind] = 1

D11 = lambda1 * v1x**2 + lambda2 * v2x**2
D22 = lambda1 * v1y**2 + lambda2 * v2y**2
D12 = lambda1 * v1x * v1y + lambda2 * v2x * v2y

for i in range(100):
    plt.imsave('im_%04d.png' % i, im, cmap=plt.cm.gray, vmin=0, vmax=1)
    gx, gy = np.gradient(im)
    gxx, tmp = np.gradient(D11 * gx + D12 * gy)
    tmp, gyy = np.gradient(D12 * gx + D22 * gy)
    tv = gxx + gyy
    im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
