#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
phi = (X - 1.2)**2 * (X + 1.2)**2 + Y**2 - 4
psi = phi.copy()

dt = .05

for i in range(100):
    plt.imsave('psi_%04d.png' % i, psi, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.clf()
    plt.contour(psi, levels=[0])
    plt.savefig('cont_%04d.png' % i)
    phi = phi + 0.05
    psi = phi.copy()
    for i in range(30):
        gx, gy = np.gradient(psi)
        gradnorm = np.sqrt(gx**2 + gy**2)
        psi = psi - dt * np.sign(phi) * (gradnorm - 1)
