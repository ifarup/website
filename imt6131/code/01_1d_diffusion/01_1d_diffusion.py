#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1)
y = np.sin(2 * np.pi * x)
y = y + .3 * np.random.randn(np.shape(x)[0])
dt = .6

for i in range(100):
    plt.clf()
    plt.plot(x, y)
    plt.ylim((-1.5, 1.5))
    plt.savefig('im_%04d.png' % i)
    y[1:-1] = y[1:-1] + dt * (y[2:] - 2 * y[1:-1] + y[:-2])
