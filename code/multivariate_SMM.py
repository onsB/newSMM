# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:19:05 2020

@author: onsbo
"""

import numpy as np
import math
from scipy.special import gamma, digamma, polygamma
from scipy.spatial import distance
from scipy.linalg import cholesky
from scipy.sparse import spdiags
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score


def multi_student_pdf(x, params, k):
    '''computes student's t pdf for a vector x
    '''
    deg_k = params['degs'][k]
    sig_k = params['cov_mat'][k]
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

def gamma_weights(x, K, params):
    n,d = x.shape
    degs = params['degs']
    means = params['means']
    sig = params['cov_mat']
    gamma_w = np.zeros((n,K))
    for i in range(n):
        for k in range(K):
            gamma_w[i,k] = (degs[k] + d) / (degs[k] + distance.mahalanobis(x[i], means[k], sig[k]))
    return(gamma_w)
    
def update_weights(posterior):
    counts = posterior.sum(axis=0)
    weights = np.divide(counts, np.sum(counts))
    return (weights)

def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n).toarray()

def update_means_cov(x, params, posterior, gamma_w, k):
    n,d = x.shape
    p_k = posterior[:,k]
    g_k = gamma_w[:,k]
    p_g_k = p_k * g_k #element-wise multiplication of posteriors by gamma weights for component k   
    mu = np.dot(p_g_k.T, x) / np.dot(p_k, g_k) #mu: new mean vector for component k
    cov = (np.dot(p_g_k * (x - mu).T, (x - mu)) /  p_k.sum()) + (1e-5 * np.eye(d)) #cov: new d*d covariance matrix for component k
    return (mu, cov)


def update_dof(x, params, posterior, gamma_w, k):
    n,d = x.shape
    d_k = params['degs'][k]
    p_k = posterior[:,k]
    g_k = gamma_w[:,k]
    p_log_g = np.dot(p_k, (np.log(g_k) - g_k)) #sum of products of posterior[i,k] by (log(gammaweight[i,k]) - gammaweight[i,k]) over i
    constant = 1 - np.log((d_k + d)/2) + digamma((d_k + d)/2) + p_log_g / p_k.sum()
    
    def f(x):
        return(constant + np.log(x/2)  - digamma(x/2))
    def fprime(x):
        return(1/x - polygamma(1, x)/2)
        
    #perform newton raphson algorithm to identify the solution of the equation f(x) = 0
    new_deg = d_k #initialization
    for i in range(100):
        test = f(new_deg) / fprime(new_deg)
        if (abs(test) >= 1e-5 and new_deg > test):
            new_deg = new_deg - test
        else:
            break
    return(new_deg)
        
    
def EM(x, params):
    iteration = 0
    K = params['prior'].shape[0]
    ll = None
    ll_log = []
    thresh = 0.0001
    new_params = deepcopy(params)
    for i in range(1000):
        #expectation step
        #calculate {new loglikelihood, posterior probabilities, gamma weights}
        iteration = iteration + 1
        old_params = deepcopy(new_params)
        log_likelihood, posterior = post_lik(x, K, old_params)
        gamma_w = gamma_weights(x, K, params)

        #maximization step
        #calculate new weights, means, cov matrices, degrees of freedom
        new_params['prior'] = update_weights(posterior)
        for k in range(K):
            new_params['means'][k], new_params['cov_mat'][k] = update_means_cov(x, old_params, posterior, gamma_w, k)
            new_params['degs'][k] = update_dof(x, old_params, posterior, gamma_w, k)

        ll_log.append(log_likelihood)
        print('iter: ', iteration)
        if ll is not None and (log_likelihood - ll) < thresh and log_likelihood > -np.inf:
            ll = log_likelihood
            print('converged ' + str(log_likelihood))
            print(new_params)
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
        #cov[cov < 1e-8] = 1e-8
        covs.append(np.sqrt(cov))
        #covs.append(cov)

    params = {'prior': np.array(weights), 'means': np.array(means), 'cov_mat': np.array(covs), 'degs': np.array(degs)}
    return params

iris = datasets.load_iris()
params = init_params(iris.data, np.unique(iris.target).shape[0])
print(params)
result = EM(iris.data, params)

def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return (posterior.argmax(axis=1))

print_performance(result, iris.target)

#cancer = datasets.load_breast_cancer()

#parameters_c = init_params(cancer.data, np.unique(cancer.target).shape[0])
#results_c = EM(cancer.data, parameters_c)
#
#print_performance(results_c,cancer.target)