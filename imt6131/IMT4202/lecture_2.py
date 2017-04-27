#!/usr/bin/env python

from pylab import *
import Image

# Analog signal

t = linspace(0, 1, 400)
x = 2*cos(2*pi*5*t) + .8*sin(2*pi*12*t) + 0.3*cos(2*pi*47*t)
plot(t,x)
savefig('signal.pdf')

# Sampled signal

clf()
t = linspace(0, 1, 129)
x = 2*cos(2*pi*5*t) + .8*sin(2*pi*12*t) + 0.3*cos(2*pi*47*t)
plot(t,x,'*')
savefig('signal_sampled.pdf')

# Plot different versions of the DFT of x

t = t[0:128]                    # skip last element
x = x[0:128]

clf()
X = fft(x)
plot(abs(X))
title('Amplitude spectrum')
savefig('amplitude.pdf')

clf()
plot(arange(-64,64), abs(fftshift(X)))
title('Amplitude spectrum (centered)')
savefig('amplitude_centered.pdf')

clf()
plot(abs(X)**2/128)
title('Power spectrum')
savefig('power.pdf')

clf()
plot(arange(-64,64), abs(fftshift(X))**2/128)
title('Power spectrum (centered)')
savefig('power_centered.pdf')

# Denoised signal

clf()
t = linspace(0, 1, 400)
x = 2*cos(2*pi*5*t) + .8*sin(2*pi*12*t)
plot(t, x)
savefig('denoised_signal.pdf')

# Aliasing

clf()
t = linspace(0, 1, 64+1)
t = t[0:64]
x = 2*cos(2*pi*5*t) + .8*sin(2*pi*12*t) + 0.3*cos(2*pi*47*t)
plot(t,x,'*')
savefig('signal_sampled_64.pdf')

clf()
X = fft(x)
plot(range(-32,32),fftshift(abs(X)))
savefig('signal_fft_64.pdf')

# White noise

clf()
x = randn(500)
plot(x)
title('Random signal with $\mu = 0$ and $\sigma^2 = 1$')
savefig('white_noise.pdf')

clf()
X = fft(x)
semilogy(abs(X)**2/500)
title('Two-sided power spectral density')
savefig('white_noise_fft.pdf')

# Signal in Noise

t = arange(0, 256*256)
t = t/256.
x = 0.1*cos(2*pi*40*t) + randn(len(t))

clf()
plot(t[0:256], x[0:256])
title('Signal in Noise, 1s')
xlim(0,1)
savefig('signal_in_noise_1s.pdf')
clf()
X = fft(x[0:256])
semilogy(abs(X)**2/256)
xlim(0,256)
title('Signal in Noise 1s, power spectrum')
savefig('signal_in_noise_1s_fft.pdf')

clf()
plot(t[0:256*16], x[0:256*16])
title('Signal in Noise, 16s')
xlim(0,16)
savefig('signal_in_noise_16s.pdf')
clf()
X = fft(x[0:256*16])
semilogy(abs(X)**2/(256*16))
xlim(0,256*16)
title('Signal in Noise 16s, power spectrum')
savefig('signal_in_noise_16s_fft.pdf')

clf()
plot(t, x)
title('Signal in Noise, 256s')
xlim(0,256)
savefig('signal_in_noise_256s.pdf')
clf()
X = fft(x)
semilogy(abs(X)**2/(256*256))
xlim(0,256*256)
title('Signal in Noise 256s, power spectrum')
savefig('signal_in_noise_256s_fft.pdf')

# 2D FFT of lena

im = asarray(Image.open('lena.png')) # read and convert to array
im = double(im)/255.                 # and to double
im = im.sum(2)/3.                  # convert to grayscale

X = log(1 + abs(fftshift(fft2(im))))
X = (X - X.min())/(X.max() - X.min())
Image.fromarray(uint8(X*255)).save('lena_fft.png')
