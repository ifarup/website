#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1)
y0 = np.sin(2 * np.pi * x)
y0 = y0 + .3 * np.random.randn(np.shape(x)[0])
y = y0.copy()

dt = .02
lambd = .01

for i in range(500):
    plt.clf()
    plt.plot(x, y)
    plt.ylim((-1.5, 1.5))
    plt.savefig('im_%04d.png' % i)
    y[1:-1] = y[1:-1] + dt * (y[2:] - 2 * y[1:-1] + y[:-2]) - \
              lambd * dt * (y[1:-1] - y0[1:-1])
