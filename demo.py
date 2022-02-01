
import skimage.filters
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2gray
from skimage import feature
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import image_kmeans
import HMRF_EM
from sklearn.cluster import Birch
from numpy import unique
from numpy import where

# load image:
I = cv.imread('1.png')
Y = rgb2gray(I)

# Edge Image:
Z = feature.canny(Y, sigma=3)
#imsave('edge.png', np.uint8(Z))
#imsave('edge2.png', np.uint8(Z2*255))
imshow(Z, cmap='gray')
plt.show()
#imshow(Z2, cmap='gray')
#plt.show()

# Blur Image:
Y = Y*255
Y = cv.GaussianBlur(Y, (3, 3), 0)
#imsave('blurredImage.png', Y)
imshow(Y, cmap='gray')
plt.show()

# Set parameters:
k = 3
EM_iter = 10
MAP_iter = 10

start_time = time.clock()
print('start K-Means Segmentation...')
X, mu, sigma = image_kmeans.image_kmeans(Y, k)
#imsave('segmentedIamge.png', X)
#print(mu)
#print(sigma)
#plt.imshow(X)
#plt.show()
[X, mu, sigma] = HMRF_EM.HMRF_EM(X, Y, Z, mu, sigma, k, EM_iter, MAP_iter)
imsave('finla_labeled_Image.png', np.uint8(X*120))
#plt.imshow(X)
#plt.show()
end_time = time.clock()
print('Duration= {}'.format(end_time-start_time))
