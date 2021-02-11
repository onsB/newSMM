# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 04:12:22 2020

@author: onsbo
"""

import numpy as np
import scipy.linalg as alg
import matplotlib.pyplot as plt

# from scipy.spatial.distance import cdist

def multivariate_normal_sampler(mu, cov, n=1):
    l = alg.cholesky(cov)
    z = np.random.normal(size=(n, cov.shape[0]))
    return z.dot(l) + mu


def multivariate_t_sampler(mu, cov, dof, n=1):
    m = mu.shape[0]
    u = np.random.gamma(dof / 2., 2. / dof, size=(n, 1))
    y = multivariate_normal_sampler(np.zeros((m,)), cov, n)
    return y / np.tile(np.sqrt(u), [1, m]) + mu


#mean = np.array([-5., 2])
#cov = np.array([[1., 0.3], [0.3, 2.]])

#xx = multivariate_t_sampler(mean, cov, 4, 1000000)
#print(xx[:10])


def visualize(params, n, k):
    cov = params['cov_mat'][k]
    mu = params['means'][k]
    deg = params['degs'][k]
    y = multivariate_t_sampler(mu, cov, deg, n)
    #x = np.arange(n)
    plt.hist(y, density = True)
    plt.show
    return(y)