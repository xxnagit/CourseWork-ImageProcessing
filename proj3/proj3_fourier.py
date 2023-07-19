#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:46:36 2019
Fourier Transform in OpenCV
"""
import tkinter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import PIL.Image, PIL.ImageTk
from PIL import ImageGrab

# Create a window
window = tkinter.Tk()
window.title("Fourier Transform using Opencv")
# Load an image using OpenCV
img = cv2.imread("cityhallRe.jpg")
imgF = cv2.imread("cityhallRe.jpg",0)
imF = np.float32(imgF)
dft = cv2.dft(imF,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
global fshift

def fourierTrans(): 
    global img
    global magnitude_spectrum
   
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    #img = cv2.blur(img, (3, 3))
    #photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    #canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
class Handler:
    
    def __init__(self, w):
        self.w = w
        w.bind("<Button-1>", self.xaxis)
        #w.bind("<ButtonRelease-1>", self.yaxis)
        w.bind("<ButtonRelease-1>", self.create)


    def xaxis(self, event):
        self.x1, self.y1 = (event.x - 1), (event.y - 1)

    def yaxis(self, event):
        self.x2, self.y2 = (event.x + 1), (event.y + 1)

    def create(self, event):
        self.yaxis(event)
        self.w.create_rectangle(self.x1,self.y1,self.x2,self.y2,fill='White')

#Get the updated image edited by customer
def getter():
    x2=window.winfo_rootx()+canvas.winfo_x()
    y2=window.winfo_rooty()+canvas.winfo_y()
    x1=x2+canvas.winfo_width()
    y1=y2+canvas.winfo_height()
    print("Updated Image Saved")
    ImageGrab.grab().crop((x2,y2,x1,y1)).save("./updated.png")

def InverseDTF():
    imgEdited = cv2.imread("updated.png", 0)
    f = np.fft.fft2(imgEdited)
    fshift = np.fft.fftshift(f)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()        
# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = img.shape
# Create a canvas that can fit the above image
canvas = tkinter.Canvas(window, width = width, height = height)
canvas.config(cursor='cross')
canvas.pack(expand='yes', fill='both')
# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

# Button that lets the user blur the image
b1 =tkinter.Button(window, text="Fourier Transform", command=fourierTrans)
b1.pack(anchor=tkinter.N)

Handler(canvas)

b2=tkinter.Button(window,text="Save",command=getter)
b2.pack(anchor=tkinter.S)
b3=tkinter.Button(window,text="Inverse",command=InverseDTF)
b3.pack(anchor=tkinter.S)


# Run the window loop
window.mainloop()

