#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
phi = (X - 1.2)**2 * (X + 1.2)**2 + Y**2 - 3

dt = .1
epsilon = 0.001

for i in range(30):
    gx, gy = np.gradient(phi)
    gradnorm = np.sqrt(gx**2 + gy**2)
    phi = phi - dt * np.sign(phi) * (gradnorm - 1)

dt = 1
k = .01

for i in range(200):
    plt.imsave('phi_%04d.png' % i, phi, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.clf()
    plt.contour(phi, levels=[0])
    plt.savefig('cont_%04d.png' % i)
    gx, gy = np.gradient(phi)
    gradnorm = np.sqrt(gx**2 + gy**2) + epsilon
    gxx,tmp = np.gradient(gx / gradnorm)
    tmp, gyy = np.gradient(gy / gradnorm)
    tv = gxx + gyy
    phi = phi + dt * gradnorm * (tv - k)
