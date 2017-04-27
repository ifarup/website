#!/usr/bin/env python

from pylab import *
import scipy.ndimage as nd
import pywt

# Piecewise monochromatic signal

t = arange(0, 1, 1./256)
x = 2*sin(2*pi*39*t)
ind = (t > .5)
x[ind] = sin(2*pi*15*t[ind])

clf()
plot(t, x)
title('Signal $x(t)$')
savefig('piecewise.pdf')

X = fft(x)
clf()
plot(abs(X[t <= .5]))
title('$|X_k|$')
savefig('piecewise_power.pdf')

# Windowing the signal

N = 256
M = 32
for m in [40, 110, 200]:
    w = zeros(256)
    w[m:M+m] = 1
    clf()
    xw = x*w
    plot(t, xw)
    ylim((-2.2, 2.2))
    title('$N = ' + str(N) + '$, $M = ' + str(M) + '$, $m = ' + str(m) + '$')
    savefig('piecewise_windowed_' + str(m) + '.pdf')
    clf()
    XW = fft(xw[m:M+m])
    k = arange(M/2)
    f = k*N/M
    plot(f, abs(XW[0:M/2]))
    title('$N = ' + str(N) + '$, $M = ' + str(M) + '$, $m = ' + str(m) + '$')
    savefig('piecewise_dft_windowed_' + str(m) + '.pdf')

# Spectrogram of the signal

clf()
gray()
for NF in [16, 32, 64, 128]:
    clf()
    gray()
    specgram(x, NFFT = NF, Fs = 256, noverlap = NF/2, interpolation='nearest')
    title('$M = ' + str(NF) + '$')
    savefig('piecewise_specgram_' + str(NF) + '.pdf')

# The Haar filters

N = 256

h = zeros(N)
l = zeros(N)

l[0] = .5
l[1] = .5
h[0] = .5
h[1] = -.5

L, H = fft(l), fft(h)

clf()
plot(range(0, N/2), abs(L[0:N/2]))
xlim(0, N/2)
title('$|L_k|$')
savefig('haar_lowpass_spectrum.pdf')

clf()
plot(range(0, N/2), abs(H[0:N/2]))
xlim(0, N/2)
title('$|H_k|$')
savefig('haar_highpass_spectrum.pdf')

# Haar example

N = 256
t = arange(0,1, 1./256)
x = .5*sin(2*pi*3*t) + .5*sin(2*pi*89*t)

l = array([.5, .5])
h = array([.5, -.5])

clf()
plot(t, x)
savefig('signal.pdf')

clf()
subplot(211)
plot(t, nd.convolve(x, l, mode='wrap'))
subplot(212)
plot(t, nd.convolve(x, h, mode='wrap'))
savefig('signal_hp_lp.pdf')

# Haar example with wave and edges

clf()
N = 1024
t = arange(0, 1, 1./N)
x = sin(2*pi*10*t)
x[N/4 + 1:] = x[N/4]
x[N/2 + 1:] = .5
x[3*N/4 + 1:] = -.7
plot(t,x)
ylim((-1.2, 1.2))
savefig('haar_signal.pdf')

clf()
X = fft(x)
thresh = sort(abs(X))[N/2]
X[abs(X) < thresh] = 0
xf = real(ifft(X))
plot(t,xf)
ylim((-1.2, 1.2))
savefig('haar_signal_dft.pdf')

clf()
cA, cD = pywt.dwt(x, 'haar')
subplot(211)
plot(cA)
xlim(0,N/2)
subplot(212)
plot(cD)
xlim(0,N/2)
savefig('haar_signal_cA_cD.pdf')

clf()
xd = pywt.idwt(cA, zeros(shape(cD)), 'haar')
plot(t, xd)
ylim((-1.2, 1.2))
savefig('haar_signal_approx.pdf')
