#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1)
y = np.zeros(np.shape(x))
y[25:] = 1

dt = .4

for i in range(500):
    if (i % 10 == 0):
        plt.clf()
        plt.plot(x, y)
        plt.ylim((-.1, 1.1))
        plt.savefig('im_%04d.png' % i)
    y[1:-1] = y[1:-1] + dt * (y[2:] - 2 * y[1:-1] + y[:-2])
    # y[0] = y[1]                 # Neumann
    # y[-1] = y[-2]               # conditions
