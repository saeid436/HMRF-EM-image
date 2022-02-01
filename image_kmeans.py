#  kmeans algorithm for an image
#   _image: 2D image
#   _numberOfClusters: number of clusters
#   segmented_image: 2D labeled image
#   mu: vector of means of clusters
#   sigma: vector of standard deviations of clusters
import cv2 as cv
import numpy as np

def image_kmeans(_image, _numOfClusters):

    vector_image = _image.reshape(-1, 1)
    vector_image = np.float32(vector_image)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv.kmeans(vector_image, _numOfClusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(_image.shape)

    mu = np.zeros((_numOfClusters, 1))
    sigma = np.zeros((_numOfClusters, 1))
    for i in range(1, _numOfClusters):
        yy = vector_image[labels == i]
        mu[i] = np.mean(yy)
        sigma[i] = np.std(yy)

    return segmented_image, mu, sigma
