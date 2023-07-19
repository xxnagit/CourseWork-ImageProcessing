#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:27:58 2019

Exam Programs
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Task3-a.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

#plt.hist(img.ravel(),256,[0,256])
plt.plot(hist)
plt.xlim([0,256])
plt.show()

valsNonZero = np.nonzero(hist)[0]
print(valsNonZero)

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

imgMapped = cdf[img]
plt.hist(imgMapped.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()

histM, binsM = np.histogram(imgMapped.flatten(),256,[0,256])
cdfM = histM.cumsum()
cdf_normalizedM = cdfM * histM.max()/ cdfM.max()
plt.plot(cdf_normalizedM, color = 'b')
plt.hist(imgMapped.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdfM','histogramM'), loc = 'upper left')
plt.show()

cv2.imwrite('image2.jpg',imgMapped)
img2 = cv2.imread('image2.jpg',0)
hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])
cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()
plt.plot(cdf_normalized2, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf2','histogram2'), loc = 'upper left')
plt.show()

numpyImgH = np.hstack((img,imgMapped))
cv2.imshow('Mapping',numpyImgH)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)