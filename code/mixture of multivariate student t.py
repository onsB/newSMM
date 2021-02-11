# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:58:19 2020

@author: onsbo
"""

import numpy as np
from scipy.special import gamma
from scipy.spatial import distance
import math
from scipy.sparse import spdiags
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score
import scipy.linalg
from scipy.linalg import cholesky
import scipy.special
import scipy.optimize

def chi_sqrd(x, deg):
    '''computes chi squared distribution pdf with deg degrees of freedom on array x'''
    f = np.multiply(np.power(x,(deg/2-1)), np.exp(x/2))
    return(np.divide(f,(2**(deg/2)*gamma(deg/2))))
    
def multi_student_pdf(x, params, k):
    '''computes student's t pdf for a vector x
    '''
    deg_k = params['degs'][k]
    sig_k = params['standard deviation'][k]
    sig_k = cholesky(sig_k)
    cov_det = np.power(np.prod(np.diagonal(sig_k)), 2)
    mu_k = params['means'][k]
    n_samples, dim = x.shape
    pdf = np.zeros(x.shape[0],)
    for i in range(n_samples):
        x_i = x[i]    
        num = gamma((deg_k + dim)/2)
        product = 1 + distance.mahalanobis(x_i, mu_k, sig_k)/deg_k
        denom = gamma(deg_k/2) * (deg_k * np.pi)**(dim/2) * math.sqrt(cov_det) * product ** (deg_k + dim)
        pdf[i] = np.divide(num,denom)
    return(pdf)

def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log(np.sum(np.exp(x - x_max[:, np.newaxis]), axis=1))
    else:
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=0))

def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n).toarray()


def post_lik(x, K, params):
    prior = params['prior']
    n = x.shape[0]
    logresp = np.zeros((n, K))
    for k in range(K):
        pdf = np.log(multi_student_pdf(x, params, k))
        logresp[:, k] = np.log(prior[k]) + pdf
    ll_new = np.sum(log_sum_exp(logresp, axis=1))
    logresp -= np.vstack(log_sum_exp(logresp, axis=1))
    resp = np.exp(logresp)
    return (ll_new, resp)


#update degrees of freedom
def update_dof(x, resp, params, tol=1e-4, n_iter=1000):
    """Solves the equation to calculate the next value of the degrees of freedom.
    """
    degs = params['degs']
    u = params['prior']
    n_components = degs.shape[0]
    n_dim = x.shape[1]
    resp_sum = resp.sum(axis = 0)
    
    # Initialisation
    new_degs = np.empty_like(degs)

    # Calculate the constant part of the equation to calculate the 
    # new degrees of freedom
    vdim = (degs + n_dim) / 2.0
    resp_logu_sum = np.sum(resp * (np.log(u) - u), axis=0)
    constant_part = 1.0               \
        + resp_logu_sum / resp_sum           \
        + scipy.special.digamma(vdim) \
        - np.log(vdim)

    # Solve the equation numerically using Newton-Raphson for each 
    # component of the mixture
    for c in range(n_components):
        def func(x): return np.log(x / 2.0)  \
            - scipy.special.digamma(x / 2.0) \
            + constant_part[c]

        def fprime(x): return 1.0 / x \
            - scipy.special.polygamma(1, x / 2.0) / 2.0

        def fprime2(x): return - 1.0 / (x * x) \
            - scipy.special.polygamma(2, x / 2.0) / 4.0
        new_degs[c] = scipy.optimize.newton(
            func, degs[c], fprime, args=(), tol=tol, 
            maxiter=n_iter, fprime2=fprime2
        )
    return new_degs

#update weights
def update_weights(posterior):
    counts = posterior.sum(axis=0)
    weights = np.divide(counts, np.sum(counts))
    return weights


def update_sd(x,posterior,params,d):
    py_x = posterior[:,d]
    mu_k = params['means'][d]
    A= (np.matmul(diag(py_x) ,np.power( (x-mu_k) ,2)) ).sum(axis=0)
    return np.sqrt(np.divide(A,py_x.sum(axis=0)) )+ 1E-5


def update_means(x,posterior,params,d):
    py_x = posterior[:,d]
    A1 = (np.matmul(diag(py_x),x) ).sum(axis=0)
    A = np.divide(A1,py_x.sum(axis=0))
    return A

def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return (posterior.argmax(axis=1))



def demo_EM(data_set, params, K):
    ll = None
    ll_log = []
    thresh = 0.0001
    new_params = deepcopy(params)
    for i in range(1000):
        old_params = deepcopy(new_params)
        log_likelihood, posterior = post_lik(data_set, K, old_params)

        new_params['prior'] = update_weights(posterior)
        new_params['degs'] = update_dof(data_set, posterior, old_params)
        for k in range(K):
            #update mu, sigma
            new_params['means'][k] = update_means(data_set, posterior, old_params, k)
            new_params['standard deviation'][k] = update_sd(data_set, posterior, old_params, k)
    
        ll_log.append(log_likelihood)
        if ll is not None and (log_likelihood - ll) < thresh and log_likelihood > -np.inf:
            ll = log_likelihood
            print('converged ' + str(log_likelihood))
            break
        else:
            ll = log_likelihood
            print(log_likelihood)
    return posterior

def init_params(data, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
    kmeans_model.fit(data)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    in_labels = np.unique(cluster_assignment, return_index=True)[1]
    unsorted_labels = [cluster_assignment[index] for index in sorted(in_labels)]
    means = []
    for i in unsorted_labels:
        means.append([centroid for centroid in centroids][i])
    num_docs = data.shape[0]
    weights = []
    degs = []

    for i in unsorted_labels:
        # Compute the number of data points assigned to cluster i:
        num_assigned = cluster_assignment[cluster_assignment == i].shape[0]
        degs.append(num_assigned - 1)
        w = float(num_assigned) / num_docs
        weights.append(w)

    # initialize covs
    covs = []

    for i in unsorted_labels:
        #calculate ovariance matrix fr the ith component
        m_r = data[cluster_assignment == i] #member rows
        m_r = m_r - m_r.mean(0)
        cov = np.matmul(np.transpose(m_r),m_r)/m_r.shape[0]
        covs.append(np.sqrt(cov))
        # covs.append(cov)

    params = {'prior': np.array(weights), 'means': np.array(means), 'standard deviation': np.array(covs), 'degs': np.array(degs)}
    return params


iris = datasets.load_iris()

parameters = init_params(iris.data, np.unique(iris.target).shape[0])
log_likelihood, posterior = post_lik(iris.data, np.unique(iris.target).shape[0], parameters)
deg = update_dof(iris.data, posterior, parameters)
#resulting_post = demo_EM(iris.data, parameters, np.unique(iris.target).shape[0])
#
#print_performance(resulting_post,iris.target)