#!/usr/bin/env python

from pylab import *
import pywt
import Image

# Frequency response of the Haar analysis filters

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
title('Haar $|L_k|$')
savefig('haar_lowpass_spectrum.pdf')

clf()
plot(range(0, N/2), abs(H[0:N/2]))
xlim(0, N/2)
title('Haar $|H_k|$')
savefig('haar_highpass_spectrum.pdf')

# Frequency response of the Le Gall 5/3 analysis filters

N = 256

h = zeros(N)
l = zeros(N)

l[0:5] = [-1./8, 1./4, 3./4, 1./4, -1./8]
h[0:3] = [-.5, 1, -.5]

L, H = fft(l), fft(h)

clf()
plot(range(0, N/2), abs(L[0:N/2]))
xlim(0, N/2)
title('Le Gall 5/3 $|L_k|$')
savefig('legall53_lowpass_spectrum.pdf')

clf()
plot(range(0, N/2), abs(H[0:N/2]))
xlim(0, N/2)
title('Le Gall 5/3 $|H_k|$')
savefig('legall53_highpass_spectrum.pdf')

# Signal compression with different wavelets

clf()
N = 1024
t = arange(0, 1, 1./N)
x = sin(2*pi*10*t)
x[N/4 + 1:] = x[N/4]
x[N/2 + 1:] = .5
x[3*N/4 + 1:] = -.7
plot(t,x)
ylim((-1.2, 1.2))
savefig('orig_signal.pdf')

def approx(x, wavelet, level):
    ca = pywt.wavedec(x, wavelet, level=level)
    ca = ca[0]
    return pywt.upcoef('a', ca, wavelet, level, take=len(x))

for wavelet in ['haar', 'db2', 'bior3.3']:
    for level in range(1,5):
        clf()
        xd = approx(x, wavelet, level)
        plot(t, xd)
        ylim(-1.2, 1.2)
        savefig(wavelet + '_signal_' + str(level) + '_level_approx.pdf')

# Multistage signal example

clf()
subplot(411)
plot(x)
xlim(0,len(x))
ylim((-3,3))
subplot(412)
plot(concatenate(pywt.wavedec(x, 'haar', level=1)))
xlim(0,len(x))
ylim((-3,3))
subplot(413)
plot(concatenate(pywt.wavedec(x, 'haar', level=2)))
xlim(0,len(x))
ylim((-3,3))
subplot(414)
plot(concatenate(pywt.wavedec(x, 'haar', level=3)))
xlim(0,len(x))
ylim((-3,3))
savefig('signal_dwt_levels.pdf')

# 2D DWT

im = (double(asarray(Image.open('lena.png')))/255).sum(2)/3. # Grayscale lena

def combine(co):
    if type(co) != list:
        return co
    else:
        Wim = concatenate((concatenate((co[0], co[1][0])),
                           concatenate((co[1][1], co[1][2]))), axis = 1)
        if len(co) == 2:
            return Wim
        else:
            return combine([Wim] + co[2:])

for level in range(1, 5):
    co = pywt.wavedec2(im, 'haar', level=level)
    Wim = combine(co)
    Wim = Wim - Wim.min()
    Wim = Wim/Wim.max()
    Image.fromarray(uint8(255*Wim)).save('lena_dwt_%d.png' % level)
