#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histgram Matching
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
#Initiate bin size of each dimension
hBins = 256
sBins = 256
#Resize the original files since they are too big.
#indoor = cv2.imread('indoor.jpg')
#shaded = cv2.imread('shaded.jpg')
#sunlit = cv2.imread('sunlit.jpg')

#indoorRe = cv2.resize(indoor, (200,200))
#shadedRe = cv2.resize(shaded, (200,200))
#sunlitRe = cv2.resize(sunlit, (200,200))

#cv2.imwrite('indoorRe.jpg', indoorRe)
#cv2.imwrite('shadedRe.jpg', shadedRe)
#cv2.imwrite('sunlitRe.jpg', sunlitRe)

#Read in the images
indoorRe = cv2.imread('indoorRe.jpg')
shadedRe = cv2.imread('shadedRe.jpg')
sunlitRe = cv2.imread('sunlitRe.jpg')

#Change the BGR to LAB and HSV color space
indoorLab = cv2.cvtColor(indoorRe, cv2.COLOR_BGR2LAB)
shadedLab = cv2.cvtColor(shadedRe, cv2.COLOR_BGR2LAB)
sunlitLab = cv2.cvtColor(sunlitRe, cv2.COLOR_BGR2LAB)

indoorHsv = cv2.cvtColor(indoorRe, cv2.COLOR_BGR2HSV)
shadedHsv = cv2.cvtColor(shadedRe, cv2.COLOR_BGR2HSV)
sunlitHsv = cv2.cvtColor(sunlitRe, cv2.COLOR_BGR2HSV)

#print(sunlitLab.shape) #check the channels
#Calculate the histogram
channels = [0, 1]
histSize = [hBins, sBins]
#hue and saturation
hRanges = [0, 180]
sRanges = [0, 256]
ranges = hRanges + sRanges

histIn = cv2.calcHist([indoorHsv],channels, None, histSize, ranges)
histSh = cv2.calcHist([shadedHsv],channels, None, histSize, ranges)
histSun = cv2.calcHist([sunlitHsv],channels, None, histSize, ranges)

#Normalize the calculated histogram
cv2.normalize(histIn,histIn, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(histSh,histSh, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(histSun,histSun, 0, 1, cv2.NORM_MINMAX)

#range(4):Correlation, Chi-square, Intersection, Bhattacharyya
for cpMethod in range(4):
    in_in = cv2.compareHist(histIn, histIn, cpMethod)
    in_shaded = cv2.compareHist(histIn, histSh, cpMethod)
    in_sunlit = cv2.compareHist(histIn, histSun, cpMethod)
    shaded_sunlit = cv2.compareHist(histSh, histSun, cpMethod)
    print('Method:', cpMethod, 'In-In / In-Shaded / In-Sunlit / Shaded-Sunlit :',\
          end = '\n')
    print(in_in, '/', in_shaded, '/', in_sunlit, '/', shaded_sunlit)

#show the histogram in 2D
plt.subplot(1,3,1)
plt.imshow(histIn,interpolation = 'nearest')
plt.subplot(1,3,2)
plt.imshow(histSh,interpolation = 'nearest')
plt.subplot(1,3,3)
plt.imshow(histSun,interpolation = 'nearest')
plt.show()
#print separately or stack vertically
reHistIn = cv2.resize(histIn, (200,200), interpolation = cv2.INTER_AREA)
#cv2.imshow('Histogram Indoor', reHistIn)
reHistSh = cv2.resize(histSh, (200,200), interpolation = cv2.INTER_AREA)
#cv2.imshow('Histogram Shaded', reHistSh)
reHistSun = cv2.resize(histSun, (200,200), interpolation = cv2.INTER_AREA)
#cv2.imshow('Histogram Sunlit', reHistSun)
numpyImgV = np.vstack((indoorRe, shadedRe, sunlitRe))
cv2.imshow('Images Indoor/Shaded/Sunlit', numpyImgV)
numpyHistV = np.vstack((reHistIn, reHistSh, reHistSun))
cv2.imshow('Histogram Indoor/Shaded/Sunlit', numpyHistV)
cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)

#test of pring the histogram
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([indoor],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.show()
