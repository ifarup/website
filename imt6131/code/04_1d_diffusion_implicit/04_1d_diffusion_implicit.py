#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

x = np.linspace(0, 1)
N = np.shape(x)[0]
y = np.sin(2 * np.pi * x)
y = y + .3 * np.random.randn(np.shape(x)[0])

dt = 2

upperdiag = np.concatenate(([0,0], -dt * np.ones(N - 2)))
centerdiag = np.concatenate(([1], (1 + 2 * dt) * np.ones(N - 2), [1]))
lowerdiag = np.concatenate((-dt * np.ones(N - 2), [0,0]))
diags = np.array([upperdiag, centerdiag, lowerdiag])
A = spdiags(diags, [1,0,-1], N, N).todense()

for i in range(100):
    plt.clf()
    plt.plot(x, y)
    plt.ylim(-1.5, 1.5)
    plt.savefig('im_%04d.png' % i)
    y = np.linalg.solve(A, y)
 
