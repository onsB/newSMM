import pandas as pd

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import math
from scipy.special import gamma, factorial, digamma
from scipy.special import polygamma as pgamma
from sklearn import datasets
from scipy.stats import gennorm
from sklearn.metrics import accuracy_score
from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans


def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n).toarray()


def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log(np.sum(np.exp(x - x_max[:, np.newaxis]), axis=1))
    else:
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=0))


def compute_pdf(x, params, k):
    sig_k = params['standard deviation'][k]
    mu_k = params['means'][k]
    A1 = -1 * np.divide(np.power((x - mu_k), 2), 2 * np.power(sig_k, 2))

    A = np.multiply(np.divide(1, sig_k * np.sqrt(math.pi * 2)), np.exp(A1))

    return A + 1E-12


def post_lik(x, K, params):
    prior = params['prior']
    n = x.shape[0]
    logresp = np.zeros((n, K))
    for k in range(K):
        pdf = (np.log(compute_pdf(x, params, k))).sum(axis=1)
        logresp[:, k] = np.log(prior[k]) + pdf
    ll_new = np.sum(log_sum_exp(logresp, axis=1))
    logresp -= np.vstack(log_sum_exp(logresp, axis=1))
    resp = np.exp(logresp)

    return (ll_new, resp)


def update_weights(posterior):
    counts = posterior.sum(axis=0)
    weights = np.divide(counts, np.sum(counts))
    return weights

def update_sd(x,posterior,params,d):
    py_x = posterior[:,d]
    mu_k = params['means'][d]
    A= (np.matmul(diag(py_x) ,np.power( (x-mu_k) ,2)) ).sum(axis=0)
    return np.sqrt(np.divide(A,py_x.sum(axis=0)) )+ 1E-5
    # A1= np.power(x,2) - 2 * np.multiply(x,mu_k) + np.power(mu_k,2)
    # A= np.matmul(diag(py_x), A1),axis=0
    # x= csr_matrix(x2)
    # A= (x.multiply(x) - 2 * x.dot(diag(mu_k))).sum(axis=0).A1 / \
    # x.shape[
    #     0] \
    # + mu_k ** 2


    # return A + 1E-5

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
        for k in range(K):  # dont forget to change the range to K
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

    for i in unsorted_labels:
        # Compute the number of data points assigned to cluster i:
        num_assigned = cluster_assignment[cluster_assignment == i].shape[0]
        w = float(num_assigned) / num_docs
        weights.append(w)

    # initialize covs
    covs = []

    for i in unsorted_labels:
        member_rows = csr_matrix(data[cluster_assignment == i])
        cov = (member_rows.multiply(member_rows) - 2 * member_rows.dot(diag(means[i]))).sum(axis=0).A1 / \
              member_rows.shape[
                  0] \
              + means[i] ** 2
        cov[cov < 1e-8] = 1e-8
        covs.append(np.sqrt(cov))
        # covs.append(cov)

    params = {'prior': np.array(weights), 'means': np.array(means), 'standard deviation': np.array(covs)}
    return params


iris = datasets.load_iris()

parameters = init_params(iris.data, np.unique(iris.target).shape[0])
resulting_post = demo_EM(iris.data, parameters, np.unique(iris.target).shape[0])
#
print_performance(resulting_post,iris.target)

