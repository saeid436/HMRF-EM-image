#  The MAP algorithm
#---input---------------------------------------------------------
#   X: initial 2D labels
#   Y: image
#   Z: 2D constraints
#   mu: vector of means
#   sigma: vector of standard deviations
#   k: number of labels
#   MAP_iter: maximum number of iterations of the MAP algorithm
#   show_plot: 1 for showing a plot of energy in each iteration
#       and 0 for not showing
#---output--------------------------------------------------------
#   X: final 2D labels
#   sum_U: final energy

import cv2 as cv
import numpy as np
import ind2ij
import matplotlib.pyplot as plt

def MRF_MAP(X, Y, Z, mu, sigma, k, MAP_iter, show_plot):
    m, n = Y.shape
    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    U = np.zeros((m*n, k))
    sum_U_MAP = np.zeros((1, MAP_iter))
    for it in range(0, MAP_iter):   # iterations
        print('Inner iteration: \n', it)
        U1 = U.copy()
        U2 = U.copy()

        for l in range(0, k):  # all labels
            yi = y - mu[l]
            temp1 = np.nan_to_num((yi*yi) / (sigma[l]**2), nan=0, posinf=255) /2
            temp1 = temp1 + np.nan_to_num(np.log(sigma[l]), nan=0, neginf=0, posinf=255)
            U1[:, [l]] = U1[:, [l]]+temp1

            for ind in range(0, m*n):  # all pixels
                i, j = ind2ij.ind2ij(ind, m)
                u2 = 0
                if (i-1 >= 0) and (Z[i-1, j] == 0):
                    u2 = u2 + (l != X[i-1, j]) / 2
                if (i+1 <= m-1) and (Z[i+1, j] == 0):
                    u2 = u2 + (l != X[i+1, j]) / 2
                if (j-1 >= 0) and (Z[i, j-1] == 0):
                    u2 = u2 + (l != X[i, j-1]) / 2
                if (j+1 <= n-1) and (Z[i, j+1] == 0):
                    u2 = u2 + (l != X[i, j+1]) / 2
                U2[ind, l] = u2
        U = U1 + U2
        temp = np.min(U, axis=1)
        x = np.argmin(U, axis=1)
        sum_U_MAP[0][it] = np.sum(temp)

        X = np.reshape(x, (m, n))
        if (it >= 3) and (np.std(sum_U_MAP[0][it-3: it])/sum_U_MAP[0][it] < 0.0001):
            break
    sum_U = 0
    for ind in range(0, (m * n)):  # all pixels
        sum_U = sum_U + U[ind, x[ind]]

    if show_plot == 1:
        t = range(0, it)
        plt.plot(t, sum_U_MAP[0: it], color='r')
        plt.title('sum U MAP')
        plt.xlabel('MAP iteration')
        plt.ylabel('sum U MAP')
        plt.show()
    return X, sum_U
