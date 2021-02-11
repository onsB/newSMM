import numpy as np
from sklearn import datasets
import bounded_multivariate_SMM as smm
from copy import deepcopy


'''
def em_debug(x, params):
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
        log_likelihood, posterior = smm.post_lik(x, K, new_params)

        # maximization step
        # calculate new weights, means, cov matrices, degrees of freedom
        new_params['prior'] = smm.update_weights(posterior)
        # update bounds before beginning maximization

        for k in range(K):
            m = 50000  # sample size for data generation
            s, h_s = smm.sample(new_params, k, m)  # data generation
            Rk, Gk = smm.R(new_params, s, h_s,
                       k)  # preparing terms for the calculation of the new mean and covariance matrix
            new_params['means'][k], new_params['cov_mat'][k] = smm.update_means_cov(x, new_params, posterior, Rk, Gk, k)

            new_params['degs'][k] = smm.update_dof(x, new_params, posterior, s, h_s, k)
        smm._validate_covariances(new_params['cov_mat'], K)
        # new_params['lower_bound'],new_params['upper_bound'] = update_bounds(x, posterior, K)
        # print('iter: ', i, 'cov: ', new_params['cov_mat'])
        new_params['lower_bound'], new_params['upper_bound'] = smm.update_bounds(x, new_params, posterior, K)
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
'''

iris = datasets.load_iris()
x = iris.data
params = smm.init_params(iris.data, np.unique(iris.target).shape[0])
K = params['prior'].shape[0]
log_likelihood, posterior = smm.post_lik(x, K, params)
for k in range(K):
    m = 100000  # sample size for data generation
    s, h_s = smm.sample(params, k, m)  # data generation
    Rk, Gk = smm.R(params, s, h_s,
                   k)  # preparing terms for the calculation of the new mean and covariance matrix
    print('Gk: ', Gk)
    means, covars = smm.update_means_cov(x, params, posterior, Rk, Gk, k)

    #new_params['degs'][k] = smm.update_dof(x, new_params, posterior, s, h_s, k)
#smm._validate_covariances(covars, K)

def R(params, s, h_s, k):
    m = 100000
    mu_k = params['means'][k]
    d = mu_k.shape[0]
    sig_k = params['cov_mat'][k]

    ##s = data_sampler.multivariate_t_sampler(mu_k, sig_k, d_k, m)

    # calculate term R for means update
    ##h_s = np.array([ h(x, params, k) for x in s])
    ind = smm.indicate(s, params, k)
    ind_sum = ind.sum()
    Rk = ((s - mu_k).T * h_s).T * ind.reshape((m, 1))
    Rk = Rk.sum(axis=0) / ind_sum

    # calculate term G for covariance matrix update
    Gk = np.array(
        [sig_k - x.reshape((d, 1)) * (x.reshape((d, 1)).T) * h(x, params, k) for x in s])  # array of n (d*d) matrices
    Gk = (Gk * ind.reshape((m, 1, 1))).sum(axis=0) / ind_sum
    return (Rk, Gk)