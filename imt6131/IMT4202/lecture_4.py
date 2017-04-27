# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/ivarf/.spyder2/.temp.py
"""

from pylab import *
import Image
import scipy.ndimage as nd

# Filtering in the frequency domain

t = arange(0, 1, 1/128.)
x  = 2*cos(2*pi*5*t) + 0.8*sin(2*pi*12*t) + 0.3*cos(2*pi*47*t)

clf()
plot(t,x, '-')
title('Sampled, linearly interpolated signal')
savefig('sampled_signal.pdf')

clf()
X = fft(x)
plot(2*abs(X[0:64])**2/128)
ylim(0,300)
title('Single sided power spectrum of original signal')
savefig('power_signal.pdf')

clf()
X[40:-40] = 0
plot(2*abs(X[0:64])**2/128)
ylim(0,300)
title('Single sided power spectrum of DFT filtered signal')
savefig('power_filtered_signal.pdf')

clf()
xt = real(ifft(X))
plot(t,xt,'-')
title('Sampled, DFT filtered, linearly interpolated signal')
savefig('sampled_filtered_signal.pdf')

clf()
f=array([3,2,1])/6.
xf = convolve(x,f,'same')
plot(t,xf,'-')
title('Sampled, convolution filtered, linearly interpolated signal')
savefig('sampled_convoluted_signal.pdf')

clf()
X = fft(xf)
plot(2*abs(X[0:64])**2/128)
ylim(0,300)
title('Single sided power spectrum of convolution filtered signal')
savefig('power_convoluted_signal.pdf')

# The lowpass filter

clf()
l = zeros(128)
l[0:3] = array([3,2,1])/6.
L = fft(l)
plot(range(-64,64),abs(fftshift(L)))
xlim(-64,64)
title('Frequency response of low-pass filter')
savefig('frequency_response_lp.pdf')

clf()
plot(range(-64,64),angle(fftshift(L)))
xlim(-64,64)
title('Phase response of low-pass filter')
savefig('phase_response_lp.pdf')

# Ideal high-pass filter

clf()
H = zeros(128)
H[32:97] = 1
plot(H)
ylim(-0.2, 1.2)
title('$H_k$')
savefig('high_pass_dft.pdf')

clf()
h = ifft(H)
plot(real(h))
xlim(0,127)
title('$h_n$')
savefig('high_pass.pdf')

# Not so ideal high-pass filter

for epsilon in [.01, .05, .1]:
    clf()
    h[abs(h) < epsilon] = 0
    plot(real(h))
    xlim(0,127)
    title('$h_{' + str(epsilon) + '}$')
    savefig('high_pass_' + str(int(100*epsilon)) + '.pdf')
    
    clf()
    H = fft(h)
    plot(abs(H))
    xlim(0,127)
    title('$H_{' + str(epsilon) + '}$')
    savefig('high_pass_' + str(int(100*epsilon)) + '_dft.pdf')
    
# 2D Filtering

D = zeros((100,100))
D[0:3,0:3] = 1/9.
Df = fft2(D)
Image.fromarray(uint8(255*abs(fftshift(Df)))).save('2D_filter.png')

# Noising and denoising Lena

im = double(asarray(Image.open('lena.png')))/255 # Grayscale lena
im = im.sum(2)/3.
Image.fromarray(uint8(255*im)).save('lena_gray.png')

imn = im + 0.05*randn(shape(im)[0], shape(im)[1])
imn[imn < 0] = 0
imn[imn > 1] = 1
Image.fromarray(uint8(255*imn)).save('lena_noise.png')

f = ones((3,3)) # averaging filter
f = f/f.sum()

imf = nd.convolve(imn, f) # filter once
Image.fromarray(uint8(255*imf)).save('lena_denoise_1.png')

for i in range(5):
    imf = nd.convolve(imf, f) # filer five more times
Image.fromarray(uint8(255*imf)).save('lena_denoise_N.png')

# Edge detection of Lena

fh = array([[1], [-1]]) # horizontal edge filter
him = abs(nd.convolve(im,fh))
him = 1 - (him - him.min()/(him.max() - him.min()))
Image.fromarray(uint8(255*him)).save('lena_h_edge.png')

fv = array([[1, -1]]) # vertical edge filter
vim = abs(nd.convolve(im, fv))
vim = 1 - (vim - vim.min()/(vim.max() - vim.min()))
Image.fromarray(uint8(255*vim)).save('lena_v_edge.png')

eim = sqrt(vim**2 + him**2) # combine edges
eim = (eim - eim.min())/(eim.max() - eim.min())
Image.fromarray(uint8(255*eim)).save('lena_edge.png')
