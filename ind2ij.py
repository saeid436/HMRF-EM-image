#  index to i and j conversion
#   ind: index
#   m: height of image
#   i, j: image coordinates
import numpy as np

def ind2ij(ind, m):

    i = np.mod(ind, m)
    j = np.uint(np.floor(ind/m))
    return i, j
