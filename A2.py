import numpy as np
import cv2

#Define the window name
windowName = "Bilateral Filter"

#Read in the first test image
img1 = cv2.imread('test2.png')
imgorg=cv2.imshow("org",img1)
#Ensure the image has loaded properly

#Apply the bilateral filter
filtered1 = cv2.bilateralFilter(img1,9,75,75)

#Display the filtered image
cv2.imshow(windowName,filtered1)
#Close the window when the user presses any key
cv2.waitKey(0)
cv2.destroyWindow(windowName)
imgorg=cv2.imshow("BF",filtered1)




"""
import numpy as np
import cv2
import math
from scipy.signal import convolve2d
img = cv2.imread("test2.png")
windowName = "Denoised Image"; # window name

cv2.imshow("Original", img);

# Creation of the noisy image

height, width, channels = img.shape
print(img.shape)
mean = 0
var = 2
sigma = var**2
gauss = np.random.normal(mean,sigma,(height, width, channels))
gauss = gauss.reshape(height, width, channels)
noisy = (img + gauss)
noisy = cv2.convertScaleAbs(noisy) # if values are higher than 255


cv2.namedWindow('Noised Image')
cv2.imshow('Noised Image',noisy)
cv2.waitKey(0)




# Bilateral Filtering - removing noise - cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
# http://people.csail.mit.edu/sparis/bf_course/slides/03_definition_bf.pdf

# Size needed for sigmaSpace = 2% of img diagonal
diag = np.sqrt((width)**2 + (height)**2)
sigmaSpace = 0.02 * diag
sigmaColor = 30
print("SS ",sigmaSpace)
print("SC ", sigmaColor)
print("image ", img.shape)

#denoisedImage = cv2.bilateralFilter(img, 9, 30 , sigmaSpace)
#denoisedImage1 = cv2.bilateralFilter(img, 9, 60 , sigmaSpace)
#denoisedImage1 = cv2.bilateralFilter(img, 9, 20, 100)

denoisedImage = cv2.bilateralFilter(img, 9, sigmaColor, sigmaSpace)


cv2.imshow("Bilateral Filter", denoisedImage);
cv2.imshow("Bilateral Filter1", denoisedImage1);
#cv2.imshow("new1", denoisedImage1);

"""


