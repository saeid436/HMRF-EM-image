#  The EM algorithm
#---input---------------------------------------------------------
#   X: initial 2D labels
#   Y: image
#   Z: 2D constraints
#   mu: initial vector of means
#   sigma: initial vector of standard deviations
#   k: number of labels
#   EM_iter: maximum number of iterations of the EM algorithm
#   MAP_iter: maximum number of iterations of the MAP algorithm
#---output--------------------------------------------------------
#   X: final 2D labels
#   mu: final vector of means
#   sigma: final vector of standard deviations

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import MRF_MAP
import ind2ij

def HMRF_EM(X, Y, Z, mu, sigma, k, EM_iter, MAP_iter):
    m, n = Y.shape
    y = Y.reshape(-1, 1)
    P_lyi = np.zeros((k, m * n))
    sum_U = np.zeros((1, EM_iter))
    for it in range(0, EM_iter):
        print('Iteration: \n', it)

        # update X
        X, sum_U[0][it] = MRF_MAP.MRF_MAP(X, Y, Z, mu, sigma, k, MAP_iter, 0)
        x = X.reshape(-1, 1)

        # update mu and sigma
        # get P_lyi
        for l in range(0, k):  # all labels
            temp1 = np.nan_to_num((1 / (np.sqrt(2 * np.pi * sigma[l]**2))), nan=0, posinf=255) * \
                    np.nan_to_num(((np.exp(-(y - mu[l])**2)) / 2 / (sigma[l]**2)), nan=0, posinf=255, neginf=0)
            temp2 = temp1 * 0
            for ind in range(0, (m * n)):  # all pixels
                i, j = ind2ij.ind2ij(ind, m)
                u = 0
                if (i-1 >= 0) and (Z[i-1, j] == 0):
                    u = u + (l != X[i-1, j])/2
                if (i+1 <= m-1) and (Z[i+1, j] == 0):
                    u = u + (l != X[i+1, j])/2
                if (j-1 >= 0) and (Z[i, j-1] == 0):
                    u = u + (l != X[i, j-1])/2
                if (j+1 <= n-1) and (Z[i, j+1] == 0):
                    u = u + (l != X[i, j+1])/2
                temp2[ind] = u
            P_lyi[l, :] = (temp1*np.exp(-temp2)).transpose()

        temp3 = np.sum(P_lyi, axis=0)
        P_lyi = np.nan_to_num(np.divide(P_lyi, temp3), nan=0)

        # get mu and sigma
        for l in range(0, k):  # all labels
            mu[l] = np.matmul(P_lyi[l, :], y)
            mu[l] = mu[l] / np.sum(P_lyi[l, :])
            sigma[l] = np.matmul(P_lyi[l, :], ((y-mu[l])**2))
            sigma[l] = sigma[l] / np.sum(P_lyi[l, :])
            sigma[l] = np.sqrt(sigma[l])

        if (it >= 2) and ((np.std(sum_U[0][it-2:it])/np.sum(sum_U[0][it])) < 0.0001):
            break

    t = range(0, it)
    plt.plot(t, sum_U[[0], 0: it].transpose(), linewidth=2)
    plt.plot(t, sum_U[[0], 0: it].transpose(), marker='.', markersize=20)
    plt.title('sum of U in each EM iteration')
    plt.xlabel('EM iteration')
    plt.ylabel('sum of U')
    plt.show()

    return X, mu, sigma