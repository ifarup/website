#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

x = np.linspace(0, 1)
N = np.shape(x)[0]
y = np.sin(2 * np.pi * x)
y = y + .3 * np.random.randn(np.shape(x)[0])

dt = .5

upperdiag = np.concatenate(([0, 0], -.5 * dt * np.ones(N - 2)))
centerdiag = np.concatenate(([1], (1 + dt) * np.ones(N - 2), [1]))
lowerdiag = np.concatenate((-.5 * dt * np.ones(N - 2), [0, 0]))
diags = np.array([upperdiag, centerdiag, lowerdiag])
A = spdiags(diags, [1, 0, -1], N, N).todense()

upperdiag = np.concatenate(([0, 0], .5 * dt * np.ones(N - 2)))
centerdiag = np.concatenate(([1], (1 - dt) * np.ones(N - 2), [1]))
lowerdiag = np.concatenate((.5 * dt * np.ones(N - 2), [0,0]))
diags = np.array([upperdiag, centerdiag, lowerdiag])
B = spdiags(diags, [1, 0, -1], N, N).todense()

y = np.array([y]).T

for i in range(100):
    plt.clf()
    plt.plot(x, y)
    plt.ylim(-1.5, 1.5)
    plt.savefig('im_%04d.png' % i)
    b = np.dot(B,y)
    y = np.linalg.solve(A, b)
