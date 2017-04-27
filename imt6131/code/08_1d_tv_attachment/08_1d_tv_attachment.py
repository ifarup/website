#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1)
y0 = np.zeros(np.shape(x))
y0[25:] = 1
y0 = y0 + .2 * np.random.randn(np.shape(x)[0])
y = y0.copy()

dt = .001
lambd = 1
eps = .01

for i in range(500):
    plt.clf()
    plt.plot(x, y)
    plt.ylim(-.3, 1.3)
    plt.savefig('im_%04d.png' % i)
    yd = y[1:] - y[:-1]
    ydtv = yd / (np.abs(yd) + eps)
    ydd = ydtv[1:] - ydtv[:-1]
    y[1:-1] = y[1:-1] + dt * ydd - \
              lambd * dt * (y[1:-1] - y0[1:-1])
