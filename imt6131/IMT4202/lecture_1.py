#!/usr/bin/env python

from pylab import *
import scipy.ndimage
import Image

# Image denoising 

lena = asarray(Image.open('lena.png')) # read and convert to array
lena = double(lena)/255.                 # and to double
lena = lena.sum(2)/3.                    # convert to grayscale

im = lena.copy()
r = rand(shape(im)[0],shape(im)[1])
im[r < .1] = 0
im[r > .9] = 1
Image.fromarray(uint8(im*255)).save('lena_noise.png')
im = scipy.ndimage.median_filter(im, 5)
Image.fromarray(uint8(im*255)).save('lena_restored.png')
im = lena.copy()
im = im + 0.2*randn(shape(im)[0], shape(im)[1])
im[im > 1] = 1
im[im < 0] = 0
Image.fromarray(uint8(im*255)).save('lena_gaussian.png')

# Edge detection

im = abs(scipy.ndimage.laplace(lena))
im = im - im.min()/(im.max() - im.min())
im = im**.5
Image.fromarray(uint8(im*255)).save('lena_edge.png')

# 1D (analog) signal

t = linspace(0,12,200)
x = .75*sin(3*t) + 0.5*sin(7*t)
clf()
plot(t,x)
savefig('analog_signal.pdf')

# 2D signal

x, y = meshgrid(linspace(0,1,200), linspace(0,1,200))
f = 1.5*cos(2*x)*cos(7*y) + 0.75*cos(5*x)*sin(3*y) - \
    1.3*sin(9*x)*cos(15*y) + 1.1*sin(13*x)*sin(11*y)
f = f - f.min()
f = f/f.max()
Image.fromarray(uint8(f*255)).save('analog_2d_signal.png')

# RGB planes

lenaC = asarray(Image.open('lena.png')) # read and convert to array
lenaR = lenaC[:,:,0]
lenaG = lenaC[:,:,1]
lenaB = lenaC[:,:,2]
Image.fromarray(lenaR).save('lena_R.png')
Image.fromarray(lenaG).save('lena_G.png')
Image.fromarray(lenaB).save('lena_B.png')

# Sampling

t = linspace(0,12,200)
x = .75*sin(3*t) + 0.5*sin(7*t)
clf()
plot(t,x)
ts = linspace(0,12,40)
xs = .75*sin(3*ts) + 0.5*sin(7*ts)
plot(ts, xs, 'r*')
savefig('sampling.pdf')
clf()
plot(ts, xs, 'r*')
savefig('samples.pdf')
clf()
plot(ts, xs)
savefig('reconstruction.pdf')
clf()
plot(t,x)
plot(ts, xs + .1*randn(len(xs)), 'r*')
ylim((-1.5, 1.5))
savefig('noise.pdf')

# Basic waveforms

for N in [4, 16]:
    m = arange(N)
    for k in arange(-N/2 + 1, N/2 + 1):
        clf()
        E = exp(2*pi*1j*k*m/N)
        plot(real(E), '*-')
        plot(imag(E), '*-')
        ylim((-1.2, 1.2))
        legend(('Re', 'Im'))
        title('$E_{' + str(N) + ',' + str(k) + '}$')
        savefig('wave_' + str(N) + '_' + str(k) + '.pdf')

for (p, q) in [(0,8), (8,0)]:
    print p,q


# Aliasing

t = linspace(0,1,400)
x1 = sin(4*pi*t)
x2 = sin(36*pi*t)
tr = linspace(0,1,17)
xr = sin(4*pi*tr)
clf()
plot(t,x1,t,x2,tr,xr,'r*')
savefig('aliasing.pdf')

# Quantisation (the cheap way)

im = uint8(255*lena)
Image.fromarray(uint8((im/2)*2)).save('lena_7bit.png')
Image.fromarray(uint8((im/4)*4)).save('lena_6bit.png')
Image.fromarray(uint8((im/8)*8)).save('lena_5bit.png')
Image.fromarray(uint8((im/16)*16)).save('lena_4bit.png')
Image.fromarray(uint8((im/32)*32)).save('lena_3bit.png')
Image.fromarray(uint8((im/64)*64)).save('lena_2bit.png')
Image.fromarray(uint8((im/128)*128)).save('lena_1bit.png')
