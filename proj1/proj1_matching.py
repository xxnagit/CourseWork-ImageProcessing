#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:49:36 2019

Matching using only hue
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

bins = 32
file = 'indoorRe.jpg' #training mode
#file = 'shadedRe.jpg' #test1 mode
#file = 'sunlitRe.jpg' #test2 mode
method = 0

#def Hist_Matching(bins):
    

img = cv2.imread(file)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#copy the channels of the original hsv to hue
channels = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv2.mixChannels([hsv], [hue], channels)

histSize = bins
ranges = [0, 180]
hist = cv2.calcHist([hue], [0], None, [histSize], ranges)
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
obj = cv2.calcBackProject([hue], [0], hist, ranges,1)
cv2.imshow('Hand Found', obj)
plt.hist(obj.ravel(),180,[0,180])
plt.show()
#Hist_Matching(bins)
cv2.imshow('Image', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(5)
