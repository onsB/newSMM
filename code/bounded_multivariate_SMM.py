# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:58:19 2020

@author: onsbo
"""

import data_sampler
import numpy as np
from numpy import linalg
import math
from scipy.special import gamma, digamma, polygamma
from scipy.spatial import distance
from scipy.linalg import cholesky
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score
import newton_raphson


def multi_student_pdf(x, params, k):
    '''computes student's t pdf for a vector x and component k
    '''
    deg_k = params['degs'][k]
    sig_k = params['cov_mat'][k]
    # sig_k = cholesky(sig_k)
    cov_det = np.power(np.prod(np.diagonal(sig_k)), 2)
    mu_k = params['means'][k]
    test = (sig_k[sig_k < 0].shape[0] > 0)
    n_samples, dim = x.shape
    pdf = np.zeros(x.shape[0], )
    for i in range(n_samples):
        x_i = x[i]
        num = gamma((deg_k + dim) / 2)
        mah = distance.mahalanobis(x_i, mu_k, sig_k)
        product = 1 + mah / deg_k
        denom = gamma(deg_k / 2) * (deg_k * np.pi) ** (dim / 2) * math.sqrt(cov_det) * product ** (deg_k + dim)
        pdf[i] = np.divide(num, denom)
        # if (i==66):
        #   print(x_i, num, product, denom)
    return pdf


def bounded_multi_student_pdf(x, params, k):
    """computes bounded student's t pdf for a vector x and component k
    """
    m = 50000  # size of the generated data to calculate the normalization constant
    pdf = multi_student_pdf(x, params, k)
    ind = indicate(x, params, k)
    pdf = pdf * ind
    n_const = normalization_const(params, k, m)
    pdf = pdf / n_const + 1e-5
    return pdf


def normalization_const(params, k, m):
    s, h_s = sample(params, k, m)
    ind = indicate(s, params, k)
    pdf = multi_student_pdf(s, params, k)
    return np.dot(pdf, ind)


def indicate(x, params, k):
    n, d = x.shape
    up_k = params['upper_bound'][k]
    low_k = params['lower_bound'][k]
    ret = np.array([int(np.where(((xi < up_k).all() and (xi > low_k).all()), 1, 0)) for xi in x])
    return ret


def h(xi, params, k):
    """gamma weights for the bounded version of SMM"""

    d = xi.shape[0]
    d_k = params['degs'][k]
    mu_k = params['means'][k]
    sig_k = params['cov_mat'][k]
    numer = d_k + d
    mah = distance.mahalanobis(xi, mu_k, sig_k)
    denom = d_k + mah
    return numer / denom


def sample(params, k, m):
    mu_k = params['means'][k]
    sig_k = params['cov_mat'][k]
    d_k = params['degs'][k]
    s = data_sampler.multivariate_t_sampler(mu_k, sig_k, d_k, m)
    h_s = np.array([h(x, params, k) for x in s])
    return s, h_s


def log_sum_exp(x, axis):
    """Compute the log of a sum of exponentials"""
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
        pdf = np.log(bounded_multi_student_pdf(x, params, k))
        logresp[:, k] = np.log(prior[k]) + pdf
    ll_new = np.sum(log_sum_exp(logresp, axis=1))
    logresp -= np.vstack(log_sum_exp(logresp, axis=1))
    resp = np.exp(logresp)
    return ll_new, resp


'''
def post_lik(x, K, params):
    prior = params['prior']
    n,d = x.shape
    resp = np.zeros((n, K))
    for k in range(K):
        pdf = bounded_multi_student_pdf(x, params, k)
        resp[:, k] = prior[k] * pdf
    ll_new = np.sum(resp.sum(axis=1))
    resp = resp / np.vstack(resp.sum(axis=1))
    return (ll_new, resp)
'''


def update_weights(posterior):
    counts = posterior.sum(axis=0)
    weights = np.divide(counts, np.sum(counts))
    return weights


# def diag(array):
#    n = len(array)
#    return spdiags(array, 0, n, n).toarray()


def R(params, s, h_s, k):
    m = 50000
    mu_k = params['means'][k]
    d = mu_k.shape[0]
    sig_k = params['cov_mat'][k]

    # s = data_sampler.multivariate_t_sampler(mu_k, sig_k, d_k, m)

    # calculate term R for means update
    # h_s = np.array([ h(x, params, k) for x in s])
    ind = indicate(s, params, k)
    ind_sum = ind.sum()
    Rk = ((s - mu_k).T * h_s).T * ind.reshape((m, 1))
    Rk = Rk.sum(axis=0) / ind_sum

    # calculate term G for covariance matrix update
    Gk = np.array(
        [sig_k - x.reshape((d, 1)) * x.reshape((d, 1)).T * h(x, params, k) for x in s])  # array of n (d*d) matrices
    Gk = (Gk * ind.reshape((m, 1, 1))).sum(axis=0) / ind_sum
    return Rk, Gk


def adjust_cov(cov):
    eigvals, v = np.linalg.eig(cov)
    l = np.diag(eigvals)
    l[l < 0] = 10e-6
    v[v < 0] = 10e-6
    ret = np.matmul(np.matmul(v, l), np.linalg.inv(v))
    return ret


def update_means_cov(x, params, posterior, Rk, Gk, k):
    n, d = x.shape
    mu_k = params['means'][k]
    p_k = posterior[:, k]
    h_x = np.array([h(xi, params, k) for xi in x])
    numer = (p_k.reshape((n, 1)) * (h_x.reshape((n, 1)) * x - Rk)).sum(axis=0)
    denom = np.dot(p_k, h_x)
    mu = numer / denom

    p_h_k = p_k * h_x  # element-wise multiplication of posteriors by gamma weights for component k
    cov = (np.dot(p_h_k * (x - mu_k).T, (x - mu_k)) / p_k.sum()) + (1e-5 * np.eye(d))
    test = not (not np.allclose(cov, cov.T) or np.any(linalg.eigvalsh(cov) <= 0))
    testgk = not (not np.allclose(Gk, Gk.T) or np.any(linalg.eigvalsh(Gk) <= 0))
    cov = cov - Gk  # cov: new d*d covariance matrix for component k
    test2 = not (not np.allclose(cov, cov.T) or np.any(linalg.eigvalsh(cov) <= 0))
    return (mu, cov)


def _validate_covariances(cov, n_components):
    """Do basic checks on matrix covariance sizes and values."""
    if len(cov.shape) != 3:
        raise ValueError(
            "'full' covars must have shape (n_components, " \
            + "n_dim, n_dim)"
        )
    elif cov.shape[1] != cov.shape[2]:
        raise ValueError(
            "'full' covars must have shape (n_components, " \
            + "n_dim, n_dim)"
        )
    for n, cv in enumerate(cov):
        if (not np.allclose(cv, cv.T)
                or np.any(linalg.eigvalsh(cv) <= 0)):
            raise ValueError(
                'component', n, " of 'full' covars must be " \
                                + "symmetric, positive-definite"
            )


def update_dof(x, params, posterior, s, h_s, k):
    '''
    update the degrees of freedom for component k
    parameters: x: dataset, n*d matrix 
                params: component k parameters
                posterior: n*K matrix of responsibilities, where K is the number of components
                s: generated multivariate student t data sample with respect to params[k], m*d matrix
                h_s: h(s), m*1 p array
                k: int, k-th component
    '''
    n, d = x.shape
    mu_k = params['means'][k]
    sig_k = params['cov_mat'][k]
    d_k = params['degs'][k]
    p_k = posterior[:, k]
    ind = indicate(s, params, k)
    ind_sum = ind.sum()
    p_k_sum = p_k.sum()
    mah = np.array([distance.mahalanobis(xi, mu_k, sig_k) for xi in x])

    def f(v):
        term1 = -digamma(v / 2) + np.log(v / 2) + digamma((v + d) / 2) - np.log((v + d) / 2) + 1
        h = (v + d) / (v + mah)
        term2 = np.dot(p_k, (np.log(h) - h)) / p_k_sum
        term3 = np.dot((np.log(h_s) - h_s + term1), ind) / ind_sum
        return term1 + term2 + term3

    def fprime(v):
        term1p = -polygamma(1, v / 2) / 2 + 1 / v + polygamma(1, (v + d) / 2) - 1 / (v + d)
        h = (v + d) / (v + mah)
        hprime = (mah - d) / (v + mah) ** 2
        term2p = np.dot(p_k, hprime * (h - 1)) / p_k_sum
        term3p = (ind * term1p).sum() / ind_sum
        return term1p + term2p + term3p

    # perform newton raphson algorithm to solve the equation f(v) = 0
    new_deg = d_k  # initialize
    for i in range(100):
        test = f(new_deg) / fprime(new_deg)
        if abs(test) >= 1e-5 and new_deg > test:
            new_deg = new_deg - test
        else:
            break
    print('new degree of freedom:', new_deg, 'gives: ', f(new_deg))
    new_deg_alt = newton_raphson.newton(f, fprime, d_k, 0.00001)
    print('alt degree of freedom:', new_deg_alt, 'gives: ', f(new_deg_alt))
    return new_deg


def update_bounds(x, params, posterior, K):
    n, d = x.shape
    new_cluster_assign = np.argmax(posterior, axis=1)  # choosing the most probable component for every vector xi in x
    upper = params['upper_bound']
    lower = params['lower_bound']
    for k in range(K):
        assign_k = x[new_cluster_assign == k]
        if assign_k.shape[0] > 1:
            upper[k] = np.amax(assign_k, axis=0)
            lower[k] = np.amin(assign_k, axis=0)
    return np.array(lower), np.array(upper)


def EM(x, params):
    K = params['prior'].shape[0]
    ll = None
    ll_log = []
    thresh = 0.000001
    new_params = deepcopy(params)
    for i in range(10):
        # expectation step
        # calculate {new loglikelihood, posterior probabilities, gamma weights}
        # old_params = deepcopy(new_params)
        print('iter ', i)
        log_likelihood, posterior = post_lik(x, K, new_params)

        # maximization step
        # calculate new weights, means, cov matrices, degrees of freedom
        new_params['prior'] = update_weights(posterior)
        # update bounds before beginning maximization
        new_params['lower_bound'], new_params['upper_bound'] = update_bounds(x, new_params, posterior, K)
        for k in range(K):
            m = 50000  # sample size for data generation
            s, h_s = sample(new_params, k, m)  # data generation
            Rk, Gk = R(new_params, s, h_s,
                       k)  # preparing terms for the calculation of the new mean and covariance matrix
            new_params['means'][k], new_params['cov_mat'][k] = update_means_cov(x, new_params, posterior, Rk, Gk, k)

            new_params['degs'][k] = update_dof(x, new_params, posterior, s, h_s, k)
        _validate_covariances(new_params['cov_mat'], K)
        # new_params['lower_bound'],new_params['upper_bound'] = update_bounds(x, posterior, K)
        # print('iter: ', i, 'cov: ', new_params['cov_mat'])
        ll_log.append(log_likelihood)
        if ll is not None and (log_likelihood - ll) < thresh and log_likelihood > -np.inf:
            ll = log_likelihood
            # print('converged ' + str(log_likelihood))
            # print(new_params)
            break
        else:
            ll = log_likelihood
            print('log likelihood: ', log_likelihood)
    return posterior, new_params


def init_params(data, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1)
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
        # calculate covariance matrix for the ith component
        m_r = data[cluster_assignment == i]  # member rows
        m_r = m_r - m_r.mean(0)
        # cov = np.matmul(np.transpose(m_r), m_r) / m_r.shape[0]
        # covs.append(np.sqrt(cov))
        cov = np.cov(m_r.T)
        cov[cov < 1e-6] = 1e-6
        covs.append(cov)

    lower_bound = []
    upper_bound = []
    for i in unsorted_labels:
        lower_bound.append(np.amin(data[cluster_assignment == i], axis=0))
        upper_bound.append(np.amax(data[cluster_assignment == i], axis=0))
    params = {'prior': np.array(weights), 'means': np.array(means), 'cov_mat': np.array(covs), 'degs': np.array(degs),
              'lower_bound': np.array(lower_bound), 'upper_bound': np.array(upper_bound)}
    return params


iris = datasets.load_iris()
params = init_params(iris.data, np.unique(iris.target).shape[0])

# print(params)

# sample_perClass = int(10e5)
result, new_params = EM(iris.data, params)


def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return posterior.argmax(axis=1)


print_performance(result, iris.target)
