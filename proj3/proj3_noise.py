#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:42:40 2019

Add noises
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL.Image

a = 3
b = 2

im = PIL.Image.open('cln1.gif')
im.save('clnCon.png', 'png')
img = cv2.imread('clnCon.png',0)
noisy1 = img + a * img.std() * np.random.random(img.shape)
noisy2 = img + b * img.max() * np.random.random(img.shape)

imF = np.float32(img)
dft = cv2.dft(imF,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
imF1 = np.float32(noisy1)
dft1 = cv2.dft(imF1,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)
magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))
imF2 = np.float32(noisy2)
dft2 = cv2.dft(imF2,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift2 = np.fft.fftshift(dft2)
magnitude_spectrum2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(noisy1),plt.title('Noisy1')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(noisy2),plt.title('Noisy2')
plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(magnitude_spectrum),plt.title('Original Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(magnitude_spectrum1),plt.title('Noisy1 Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(magnitude_spectrum2),plt.title('Noisy2 Spectrum')
plt.xticks([]), plt.yticks([])


plt.show()