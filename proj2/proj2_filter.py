# -*- coding: utf-8 -*-
"""
Spyder Editor

project 2
"""
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import filters
#Apply Gussian Blur for sigma = 2, 5, 10 for grey image
sigmaArr = np.array([2, 5, 10], 'i')

#img = cv2.imread('cityhall.jpg')
#imgRe = cv2.resize(img, (200,200))
#cv2.imwrite('cityhallRe.jpg', imgRe)
imgRe = cv2.imread('cityhallRe.jpg')
cv2.imshow('Original', imgRe)
im = np.array(Image.open('cityhallRe.jpg').convert('L'))
for i in sigmaArr:
    im2 = filters.gaussian_filter(imgRe,i)
    name = 'sigma = ' + str(i)
    cv2.imshow(name, im2)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)
#Implement an unsharp masking operation by blurring an image and then 
#subtracting the blurred version from the original.
#Define f weightfactor, sigma
factor = 2
sigma = 2
imgBlur = filters.gaussian_filter(imgRe,sigma)
mask = imgRe - imgBlur
unSharpen = imgRe + factor * mask
cv2.imshow('Grey Image Mask', mask)
cv2.imshow('Unsharpen Grey Image', unSharpen)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)
#Apply Gussian Blur for sigma = 2, 5, 10 for color image
#imgC = cv2.imread('colorhall.jpg')
#imgCRe = cv2.resize(imgC, (200,200))
#cv2.imwrite('colorhallRe.jpg', imgCRe)
imgCRe = cv2.imread('colorhallRe.jpg')
cv2.imshow('Color Original', imgCRe)
imC = np.array(Image.open('colorhallRe.jpg'))
imgConv = np.zeros(imC.shape)
for j in sigmaArr:
    for i in range(3):
        imgConv[:,:,i] = filters.gaussian_filter(imC[:,:,i],j)
        name = 'sigma = ' + str(j)
    imgConv = np.array(imgConv, 'uint8')
    cv2.imshow(name, imgConv)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)
#Implement an unsharp masking operation by blurring an image and then 
#subtracting the blurred version from the original.
imgCBlur = np.zeros(imC.shape)
for i in range(3):
    imgCBlur[:,:,i] = filters.gaussian_filter(imC[:,:,i],sigma)
    name = 'sigma = ' + str(sigma)
imgCBlur = np.array(imgCBlur, 'uint8')
cv2.imshow(name, imgCBlur)
maskC = imgCRe - imgCBlur
unSharpenC = imgCRe + factor * mask
cv2.imshow('Color Image Mask', maskC)
cv2.imshow('Unsharpen Color Image', unSharpenC)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)


