#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
phi = (X - 1.2)**2 * (X + 1.2)**2 + Y**2 - 4
H = (phi < 0)
bound = 1 - np.exp(-40 * phi**2)

for i in range(100):
    plt.imsave('phi_%04d.png' % i, phi, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.clf()
    plt.contour(phi, levels=[0])
    plt.savefig('cont_%04d.png' % i)
    phi = phi + .05
