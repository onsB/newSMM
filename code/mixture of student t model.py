# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:58:21 2020

@author: onsbo
"""

import numpy as np
from scipy.special import gamma
import math
from scipy.sparse import spdiags
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score
import scipy.linalg
import scipy.special
import scipy.optimize


def chi_sqrd(x, deg):
    '''computes chi squared distribution pdf with deg degrees of freedom on array x'''
    f = np.multiply(np.power(x,(deg/2-1)), np.exp(x/2))
    return(np.divide(f,(2**(deg/2)*gamma(deg/2))))
    
def multi_student_pdf(x, mu, sig, deg):
    '''computes student's t pdf for a vector x
       mu: vector of means
       sig: inner product matrix
       deg: degrees of freedom'''
    p = x.shape
    nom = gamma((deg + p)/2)
    product = 1 + (1/deg) * np.matmul(np.matmul(np.transpose(x-mu), 1/sig), (x-mu))
    denom = gamma(deg/2) * (deg * np.pi)**(p/2) * math.sqrt(np.linalg.det(sig)) * product ** ((deg+p)/2)
    return(np.divide(nom,denom))
    
def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n).toarray()


def student_pdf(x, params,k):
    deg = params['degs'][k]
    a = gamma((deg+1)/2)
    b = math.sqrt(deg*np.pi)*gamma(deg/2)
    c = np.power((1+x**2/deg),-(deg+1)/2)
    return((a*c)/b)

def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log(np.sum(np.exp(x - x_max[:, np.newaxis]), axis=1))
    else:
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=0))
    
    
def post_lik(x, K, params):
    prior = params['prior']
    n = x.shape[0]
    logresp = np.zeros((n, K))
    for k in range(K):
        pdf = (np.log(student_pdf(x, params, k))).sum(axis=1)
        logresp[:, k] = np.log(prior[k]) + pdf
    ll_new = np.sum(log_sum_exp(logresp, axis=1))
    logresp -= np.vstack(log_sum_exp(logresp, axis=1))
    resp = np.exp(logresp)
    return (ll_new, resp)


def update_weights(posterior):
    counts = posterior.sum(axis=0)
    weights = np.divide(counts, np.sum(counts))
    return weights
'''
def update_dof(x,posterior,params,k):
    py_x = posterior[:,k]
    A1 = (np.matmul(diag(py_x),x) ).sum(axis=0)
    A = np.divide(A1,py_x.sum(axis=0))
    return A
'''

def update_dof(x, posterior, params, k):
    degs = params['degs']
    n_dim = x.shape[1]
    p_sum = posterior.sum(axis = 0)
    u = np.array([parameters['prior'],] * x.shape[0])

    # Calculate the constant part of the equation to calculate the 
    # new degrees of freedom
    vdim = (degs + n_dim) / 2.0
    resp_logu_sum = np.sum(posterior * (np.log(u) - u), axis=0)
    constant_part = 1.0               \
        + resp_logu_sum / p_sum           \
        + scipy.special.digamma(vdim) \
        - np.log(vdim)

    # Solve the equation numerically using Newton-Raphson for each 
    # component of the mixture
    def func(x): return np.log(x / 2.0)  \
        - scipy.special.digamma(x / 2.0) \
        + constant_part[k]

    def fprime(x): return 1.0 / x \
        - scipy.special.polygamma(1, x / 2.0) / 2.0

    def fprime2(x): return - 1.0 / (x * x) \
        - scipy.special.polygamma(2, x / 2.0) / 4.0
    new_deg = scipy.optimize.newton(
        func, degs[k], fprime, args=(),fprime2=fprime2)
    return new_deg


def demo_EM(data_set, params, K):
    ll = None
    ll_log = []
    thresh = 0.0001
    new_params = deepcopy(params)
    for i in range(1000):
        old_params = deepcopy(new_params)
        log_likelihood, posterior = post_lik(data_set, K, old_params)

        new_params['prior'] = update_weights(posterior)
        for k in range(K):
            new_params['degs'][k] = update_dof(data_set, posterior, old_params, k)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

        ll_log.append(log_likelihood)
        if ll is not None and (log_likelihood - ll) < thresh and log_likelihood > -np.inf:
            ll = log_likelihood
            print('converged ' + str(log_likelihood))
            break
        else:
            ll = log_likelihood
            print(log_likelihood)
    return posterior  

def init(data, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
    kmeans_model.fit(data)
    cluster_assignment = kmeans_model.labels_
    in_labels = np.unique(cluster_assignment, return_index=True)[1]
    unsorted_labels = [cluster_assignment[index] for index in sorted(in_labels)]
    num_docs = data.shape[0]
    weights = []
    degs = []

    for i in unsorted_labels:
        # Compute the number of data points assigned to cluster i:
        num_assigned = cluster_assignment[cluster_assignment == i].shape[0]
        degs.append(num_assigned - 1)
        w = float(num_assigned) / num_docs
        weights.append(w)


    params = {'prior': np.array(weights), 'degs': np.array(degs)}
    return params

def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return (posterior.argmax(axis=1))

iris = datasets.load_iris()

parameters = init(iris.data, np.unique(iris.target).shape[0])

resulting_post = demo_EM(iris.data, parameters, np.unique(iris.target).shape[0])

print_performance(resulting_post,iris.target)   
