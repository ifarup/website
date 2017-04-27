#!/usr/bin/env python

from pylab import *
import scipy.fftpack as fp
import Image

# Step function

N = 256
clf()
t = arange(0, 1, 1./N)
x = zeros(shape(t))
x[t <= 0.2] = 1
plot(t,x)
ylim(-.2, 1.2)
xlabel('$t$')
ylabel('$x(t)$')
title('Step function')
savefig('step_function.pdf')

clf()
X = fft(x)
plot(arange(-N/2,N/2), fftshift(abs(X)))
xlim(-N/2, N/2)
xlabel('$k$')
ylabel('$|X_k|$')
title('DFT of step function')
savefig('step_function_fft.pdf')

clf()
C = fp.dct(x, norm='ortho')
plot(arange(N), abs(C))
xlim((0, N-1))
xlabel('$k$')
ylabel('$|C_k|$')
title('DCT of step function')
savefig('step_function_dct.pdf')

# Compress with different values for c

c_values = array([0.001, 0.01, 0.03, 0.1, 0.5]) # compression parameters
Xmin_values = zeros(shape(c_values))            # to store results
Cmin_values = zeros(shape(c_values))
P_fft_values = zeros(shape(c_values))
P_dct_values = zeros(shape(c_values))
mD_fft_values = zeros(shape(c_values))
mD_dct_values = zeros(shape(c_values))

for i in range(len(c_values)):
    Xmin_values[i] = c_values[i]*abs(X).max()
    P_fft_values[i] = sum(abs(X) > Xmin_values[i])/double(N)
    X[abs(X) < Xmin_values[i]] = 0 # threshold
    xt = ifft(X)
    mD_fft_values[i] = norm(x - xt)**2/norm(x)**2
    clf()
    plot(t, real(xt), t, x, '--')
    ylim(-.2, 1.2)
    xlabel('$t$')
    ylabel('$\tilde x(t)$')
    title('Step function DFT, $c = ' + str(c_values[i]) + '$')
    savefig('step_function_dft_' + str(int(1000*c_values[i])) + '.pdf')

    Cmin_values[i] = c_values[i]*abs(C).max()
    C[abs(C) < Cmin_values[i]] = 0 # threshold
    xt = fp.idct(C, norm='ortho')
    mD_dct_values[i] = norm(x - xt)**2/norm(x)**2
    clf()
    plot(t, real(xt), t, x, '--')
    ylim(-.2, 1.2)
    xlabel('$t$')
    ylabel('$\tilde x(t)$')
    title('Step function DCT, $c = ' + str(c_values[i]) + '$')
    savefig('step_function_dct_' + str(int(1000*c_values[i])) + '.pdf')

# print c_values
# print Xmin_values
# print P_fft_values
# print 100*mD_fft_values

# Linear function

N = 256
clf()
t = arange(0, 1, 1./N)
x = t.copy()
plot(t,x)
ylim(-.2, 1.2)
xlabel('$t$')
ylabel('$x(t)$')
title('Linear function')
savefig('linear_function.pdf')

clf()
X = fft(x)
plot(arange(-N/2,N/2), fftshift(abs(X)))
xlim(-N/2, N/2)
xlabel('$k$')
ylabel('$|X_k|$')
title('DFT of linear function')
savefig('linear_function_fft.pdf')

clf()
C = fp.dct(x, norm='ortho')
plot(arange(N), abs(C))
xlim((0, N-1))
xlabel('$k$')
ylabel('$|C_k|$')
title('DCT of linear function')
savefig('linear_function_dct.pdf')

# Compress with different values for c

c_values = array([0.001, 0.01, 0.03, 0.1, 0.5]) # compression parameters
Xmin_values = zeros(shape(c_values))            # to store results
Cmin_values = zeros(shape(c_values))
P_fft_values = zeros(shape(c_values))
P_dct_values = zeros(shape(c_values))
mD_fft_values = zeros(shape(c_values))
mD_dct_values = zeros(shape(c_values))

for i in range(len(c_values)):  # loop over the compression parameter
    Xmin_values[i] = c_values[i]*abs(X).max()
    P_fft_values[i] = sum(abs(X) > Xmin_values[i])/double(N)
    X[abs(X) < Xmin_values[i]] = 0 # threshold
    xt = ifft(X)
    mD_fft_values[i] = norm(x - xt)**2/norm(x)**2
    clf()
    plot(t, real(xt), t, x, '--')
    ylim(-.2, 1.2)
    xlabel('$t$')
    ylabel('$\tilde x(t)$')
    title('Linear function, $c = ' + str(c_values[i]) + '$')
    savefig('linear_function_dft_' + str(int(1000*c_values[i])) + '.pdf')

    Cmin_values[i] = c_values[i]*abs(C).max()
    P_dct_values[i] = sum(abs(C) > Cmin_values[i])/double(N)
    C[abs(C) < Cmin_values[i]] = 0 # threshold
    xt = fp.idct(C, norm='ortho')
    mD_dct_values[i] = norm(x - xt)**2/norm(x)**2
    clf()
    plot(t, real(xt), t, x, '--')
    ylim(-.2, 1.2)
    xlabel('$t$')
    ylabel('$\tilde x(t)$')
    title('Linear function, $c = ' + str(c_values[i]) + '$')
    savefig('linear_function_dct_' + str(int(1000*c_values[i])) + '.pdf')

# print c_values
# print Xmin_values
# print P_fft_values
# print 100*mD_fft_values

# Periodic extension

clf()
X = fft(x)
X[abs(X) < 0.03*abs(X).max()] = 0
xt = ifft(X)
xp = concatenate((x,x,x))
xtp = concatenate((xt, xt, xt))
tp = arange(-1, 2, 1./N)
plot(tp, real(xtp), tp, xp, '--')
ylim((-0.2, 1.2))
xlabel('$t$')
ylabel('$x(t)$')
savefig('linear_function_extension.pdf')

# Periodic extension, DCT version

clf()
plot(t,x)
xlim((0,2))
xlabel('$t$')
ylabel('$x(t)$')
title('Original signal')
savefig('signal.pdf')

clf()
plot(concatenate((t, t+1)), concatenate((x, x[::-1])))
xlim((0,2))
xlabel('$t$')
ylabel('$x(t)$')
title('Extended signal')
savefig('signal_extension.pdf')

# DCT basis functions

clf()
N = 256
m = arange(N)
for k in [0, 1, 2, 3, 4]:
    if k == 0:
        CNk = ones(N)/sqrt(N)
    else:
        CNk = sqrt(2./N)*cos(pi*k*(2*m + 1)/(2*N))
    plot(CNk)
xlim(0, N-1)
savefig('dct_basis.pdf')

# Folding of the lena image

im = asarray(Image.open('lena.png')) # read and convert to array
im = concatenate((im, im[::-1,:,:]), axis=0)
im = concatenate((im, im[:,::-1,:]), axis=1)
Image.fromarray(im).save('lena_folded.png')
im = concatenate((im, im[::-1,:,:]), axis=0)
im = concatenate((im, im[:,::-1,:]), axis=1)
Image.fromarray(im).save('lena_double_folded.png')
