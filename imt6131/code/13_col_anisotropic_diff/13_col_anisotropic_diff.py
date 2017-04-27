#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from colour.misc import dip, dim, djp, djm, dic, djc

dt = .25
eps = 1e-5
kappa = 1e4
tau = 1e-2

plt.ion()
im = plt.imread('lena.png')
im = im + .2 * np.random.rand(*np.shape(im))
im[im < 0] = 0
im[im > 1] = 1

# Gradient

gx = dic(im)
gy = djc(im)

# Structure tensor components

S11 = (gx**2).sum(2)
S12 = (gx * gy).sum(2)
S22 = (gy**2).sum(2)

# Eigenvalues and eigenvectors of the structure tensor

lambda1 = .5 * (S11 + S22 + np.sqrt((S11 - S22)**2 + 4 * S12**2))
lambda2 = .5 * (S11 + S22 - np.sqrt((S11 - S22)**2 + 4 * S12**2))

theta1 = .5 * np.arctan2(2 * S12, S11 - S22)
theta2 = theta1 + np.pi / 2

v1x = np.cos(theta1)
v1y = np.sin(theta1)
v2x = np.cos(theta2)
v2y = np.sin(theta2)

# Diffusion tensor

def D(lambdax):
    return 1 / (1 + kappa * lambdax**2)

def D_alt(lambdax):
    return np.exp(-lambdax / tau)

D11 = D(lambda1) * v1x**2 + D(lambda2) * v2x**2
D22 = D(lambda1) * v1y**2 + D(lambda2)* v2y**2
D12 = D(lambda1) * v1x * v1y + D(lambda2) * v2x * v2y

D11 = np.dstack((D11, D11, D11))
D12 = np.dstack((D12, D12, D12))
D22 = np.dstack((D22, D22, D22))

# Anisotropic diffusion

for i in range(100):
    plt.imsave('im_%04d.png' % i, im, cmap=plt.cm.gray, vmin=0, vmax=1)
    gx = dic(im)
    gy = djc(im)
    gxx = dic(D11 * gx + D12 * gy)
    gyy = djc(D12 * gx + D22 * gy)
    tv = gxx + gyy
    im[1:-1, 1:-1, :] = im[1:-1, 1:-1, :] + dt * tv[1:-1, 1:-1, :]
