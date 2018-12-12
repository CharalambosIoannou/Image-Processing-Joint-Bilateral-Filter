import numpy as np
import cv2
import math

def gauss(img,spatialKern, rangeKern):    
    gaussianSpatial = 1 / math.sqrt(2*math.pi* (spatialKern**2)) #gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gaussianRange= 1 / math.sqrt(2*math.pi* (rangeKern**2)) #gaussian function to calcualte the range kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    xx=-spatialKern + np.arange(2 * spatialKern + 1)
    yy=-spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx , yy )
    spatialGS = gaussianSpatial*np.exp(-(x **2 + y **2) /(2 * (gaussianSpatial**2) ) ) #calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2) 
    return matrix,spatialGS

def padImage(img,spatialKern): #pad array with mirror reflections of itself.
    img=np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    return img
    
def jointBilateralFilter(img, img1,spatialKern, rangeKern):
    h, w, ch = img.shape #get the height,width and channel of the image with no flash
    orgImg = padImage(img,spatialKern) #pad image with no flash
    secondImg = padImage(img1,spatialKern)   #pad image with flash  
    matrix,spatialGS=gauss(img,spatialKern, rangeKern) #apply gaussian function

    outputImg = np.zeros((h,w,ch), np.uint8) #create a matrix the size of the image
    summ=1
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            for i in range (0,ch): #iterate through the image's height, width and channel
                #apply the equation that is mentioned in the pdf file
                neighbourhood=secondImg[x-spatialKern : x+spatialKern+1 , y-spatialKern : y+spatialKern+1, i] #get neighbourhood of pixels
                central=secondImg[x, y, i] #get central pixel
                res = matrix[ abs(neighbourhood - central) ]  # subtract them                   
                summ=summ*res*spatialGS #multiply them with the spatial kernel
                norm = np.sum(res) #normalization term
                outputImg[x-spatialKern, y-spatialKern, i]= np.sum(res*orgImg[x-spatialKern : x+spatialKern+1, y-spatialKern : y+spatialKern+1, i]) / norm # apply full equation of JBF(img,img1)
    return outputImg



spatialKern = 30 #27
rangeKern =20  #10
img = cv2.imread('test3a.jpg')
img1=cv2.imread('test3b.jpg') #read both images, flash and no flash image
filteredimg = jointBilateralFilter(img, img1, spatialKern, rangeKern)
cv2.imshow('input', img) #show original no flash image
cv2.imshow('JointBilateralFilter',filteredimg ) #show image after joint bilateral filter is applied
