#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:11:00 2019

Distributive over multiplication
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL.Image

def fourierTrans(img):
    imF = np.float32(img)
    dft = cv2.dft(imF,flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

im1 = PIL.Image.open('stp1.gif')
im1.save('stp1Con.png', 'png')
img1 = cv2.imread('stp1Con.png',0)
im2 = PIL.Image.open('stp2.gif')
im2.save('stp2Con.png', 'png')
img2 = cv2.imread('stp2Con.png',0)
multi = img1 * img2
magMulti1 = fourierTrans(multi)

mag1 = fourierTrans(img1)
mag2 = fourierTrans(img2)
magMulti2 = mag1 * mag2

plt.subplot(231),plt.imshow(img1),plt.title('stp1')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(img2),plt.title('stp2')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(magMulti1),plt.title('Mutiplication First')
plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(mag1),plt.title('Spectrum of stp1')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(mag2),plt.title('Spectrum of stp2')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(magMulti2),plt.title('Fourier Transform First')
plt.xticks([]), plt.yticks([])

plt.show()