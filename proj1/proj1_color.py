# -*- coding: utf-8 -*-
"""
Spyder Editor
Color Model Convention
"""
import cv2
from matplotlib import pyplot as plt

indoorRe = cv2.imread('indoorRe.jpg')
indoorLab = cv2.cvtColor(indoorRe, cv2.COLOR_BGR2LAB)
indoorHsv = cv2.cvtColor(indoorRe, cv2.COLOR_BGR2HSV)
#print using matplotlib
plt.subplot(1,3,1)
plt.imshow(indoorRe)
plt.title('Indoor BGR')
plt.subplot(1,3,2)
plt.imshow(indoorLab)
plt.title('Indoor LAB')
plt.subplot(1,3,3)
plt.imshow(indoorHsv)
plt.title('Indoor HSV')
plt.show()
#show the windows separately
cv2.imshow('Indoor Original', indoorRe)
cv2.imshow('Indoor Lab', indoorLab)
cv2.imshow('Indoor HSV', indoorHsv)

cv2.waitKey(0) #press any key to close all the windows
cv2.destroyAllWindows()
cv2.waitKey(5)

