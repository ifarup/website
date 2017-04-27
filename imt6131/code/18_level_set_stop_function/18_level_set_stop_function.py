#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('cubes.png')
im = im[..., :3].sum(2)
im = im / im.max()

xd = im.shape[1]
yd = im.shape[0]
X, Y = np.meshgrid(np.linspace(-xd/2, xd/2, xd), np.linspace(-yd/2, yd/2, yd))
phi = np.sqrt(X**2 + Y**2) - 40

gx, gy = np.gradient(im)
g = np.sqrt(gx**2 + gy**2)
g = 1 / (1 + 1000 * g**2)

def my_sign(x, eps = 1e-3):
    return x / (np.abs(x) + eps)

dt = .24
for i in range(30):
    gx, gy = np.gradient(phi)
    gradnorm = np.sqrt(gx**2 + gy**2)
    phi = phi - dt * my_sign(phi) * (gradnorm - 1)

dt = .24
k = 0

for i in range(10000):
    if i % 10 == 0:
        plt.clf()
        plt.imshow(im, plt.cm.gray)
        plt.contour(phi, 'r', levels=[0])
        plt.savefig('cont_%04d.png' % i)
    gx, gy = np.gradient(phi)
    gradnorm = 1 + 100 * np.sqrt(gx**2 + gy**2)
    gxx,tmp = np.gradient(gx / gradnorm)
    tmp, gyy = np.gradient(gy / gradnorm)
    tv = gxx + gyy
    phi = phi +  dt * gradnorm * (tv - k) * g
