import numpy as np
from numpy import linalg
import sklearn
from sklearn.cluster import KMeans
import kmeans_init
import t_pdf
import data_sampler
from scipy.spatial import distance

class BSMM(sklearn.base.BaseEstimator):
    def __init__(self, X, n_components=1, n_iter=1000, min_covar=1e-6, m=2000):

        X = np.asarray(X)
        self.n, self.d = X.shape
        self.data = X.copy()
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.m = m #sample size for data generation

    def _init(self):
        ##initialization with kmeans
        params = kmeans_init.init_params(self.data, self.n_components)
        self.means = params['means']
        self.prior = params['prior']
        self.cov = params['cov_mat']
        self.degs = params['degs']
        self.low = params['lower_bound']
        self.up = params['upper_bound']

        #initialization with random parameters
        # self.means = np.array(np.random.random((self.n_components, self.d)) + np.mean(self.data))
        # self.cov = np.array([np.asmatrix(np.identity(self.d)) for i in range(self.n_components)])
        # self.prior = np.ones(self.n_components)/self.n_components
        # self.degs = np.random.randint(self.n, size=self.n_components)
        # z: latent variable: probability of each point for each distribution
        self.z = np.array(np.empty((self.n, self.n_components), dtype=float))
        # self.up = np.zeros((self.n_components, self.d)) + np.max(self.data, axis =0)
        # self.low = np.zeros((self.n_components, self.d)) + np.min(self.data, axis=0)
        self.samples = np.empty((self.n_components, self.m, self.d)) # vector of samples
        self.inds = np.empty(shape=(self.n_components, self.m)) # vector of indicators
        for k in range(self.n_components):
            self.inds[k], self.samples[k] = self.sample(k)

    def fit(self, tol = 1e-4):
        #apply EM algorithm
        self._init()
        '''
        num_iters = 0
        log_l = 1 #log likelihood
        prev_log_l = 0 #previous log likelihood
        while(log_l - prev_log_l > tol):
            prev_log_l = self.loglikelihood()
            self.E_step()
            self.M_step()
            num_iters += 1
            log_l = self.loglikelihood()
            print("Iteration %d : log-likelihood is %.6f"%(num_iters, log_l))
        '''
        log_l = None
        log_l_list = []
        for iter in range(self.n_iter):
            print("iter %d"%iter)
            self.z, temp_ll = self.E_step()
            self.M_step()
            log_l_list.append(temp_ll)
            if log_l is not None and (temp_ll-log_l<tol) and (temp_ll> -np.inf) and iter==3:
                log_l = temp_ll
                break
            else:
                log_l = temp_ll
                print('log-likelihood: ', log_l)


    def E_step(self):
        '''expectation step
        m: int, sample size for data generation'''
        # generate the data for all components
        for k in range(self.n_components):
            self.inds[k], self.samples[k] = self.sample(k)
        log_l = 0  # initialize log likelihood
        #finding z values (posteriors)
        resp = np.zeros((self.n, self.n_components))
        for i in range(self.n):
            for k in range(self.n_components):
                resp[i,k] = self.bsmm_pdf(self.data[i,:], self.means[k,:], self.cov[k,:], self.degs[k],  self.up[k], self.low[k], self.inds[k]) * self.prior[k]
                resp[i,:] = t_pdf.fill_nans(resp[i,:])

            log_l += np.log(resp[i].sum()) #equation 7 in the paper

        #resp /= resp.sum(axis = 1)
        resp = np.divide(resp, resp.sum(axis=1).reshape((self.n, 1)))
        return (resp, log_l)

    def M_step(self):
        #1- update mixing weights (prior)
        counts = self.z.sum(axis=0)
        self.prior = np.divide(counts, np.sum(counts))

        #2- update bounds (up, low)
        new_cluster_assign = np.argmax(self.z,axis=1)  # choosing the most probable component for every vector xi in x
        for k in range(self.n_components):
            assign_k = self.data[new_cluster_assign == k]
            if assign_k.shape[0] > 1:
                self.up[k] = np.amax(assign_k, axis=0)
                self.low[k] = np.amin(assign_k, axis=0)

        #3- update means and covariances (mean, cov)
        for k in range(self.n_components):
            self.update_mean_cov(k)

        print('new params')
        print('means')
        print(self.means)
        print('cov matrix')
        print(self.cov)
        print('eigenvalues of cov matrices')
        for k in range(self.n_components):
            evals, evecs = np.linalg.eigh(self.cov[k])
            print(evals)


    @staticmethod
    def bsmm_pdf(x, mean, cov, deg, up, low, ind):
        return t_pdf.bounded_multivariate_t_pdf(x, mean, cov, deg, up, low, ind)

    def sample(self, k):
        '''generate a sample of m data vectors from the t distribution with the parameters of the k-th component of the mixture
        m: int, number of vectors to sample
        k: int, number of the component to sample for

        returns: norm_const: int, sum of the indicator function vector applied on the sample
                 s: k*d np array, sample array
        '''
        s = data_sampler.multivariate_t_sampler(self.means[k], self.cov[k], self.degs[k], self.m)
        # apply indicator function on the sample --> get vector of m zeros and ones
        ind = np.array([int(np.where(((xi <= self.up[k]).all() and (xi >= self.low[k]).all()), 1, 0)) for xi in s])
        return ind, s

    def update_mean_cov(self,k):
        h_s = self.h(self.samples[k],k)

        # calculate term R for means update
        # h_s = np.array([ h(x, params, k) for x in s])
        ind_sum = self.inds[k].sum()
        # if ind_sum<=0:
        #     raise ValueError('bad value for ind_sum is: ',ind_sum,' at component ',k)
        # print("samples[k]: ", self.samples[k].shape)
        # print("means[k]: ", self.means[k].shape)
        # print("h_s: ", h_s.shape)
        Rk = ((self.samples[k] - self.means[k]).T * h_s).T * self.inds[k].reshape((self.m, 1))
        Rk = Rk.sum(axis=0) / ind_sum  ##equation 18

        # calculate term G for covariance matrix update
        Gk = np.array(
            [self.cov[k] - x.reshape((self.d, 1)) * x.reshape((self.d, 1)).T * self.h_indiv(x, k) for x in self.samples[k]])  # array of n (d*d) matrices
        Gk = (Gk * self.inds[k].reshape((self.m, 1, 1))).sum(axis=0) / ind_sum  ## equation 19
        #assert not (np.isnan(Gk).any())
        # if np.isnan(Gk).any():
        #     raise ValueError('Gk has nan values at component ', k)


        #mu_k = params['means'][k]
        #p_k = posterior[:, k]
        h_data = self.h(self.data, k)
        numer = (self.z[:,k].reshape((self.n, 1)) * (h_data.reshape((self.n, 1)) * self.data - Rk)).sum(axis=0)
        denom = np.dot(self.z[:,k], h_data)
        self.means[k] = numer / denom  ##equation 14

        p_h_k = self.z[:,k] * h_data  # element-wise multiplication of posteriors by gamma weights for component k
        cov = (np.dot(p_h_k * (self.data - self.means[k]).T, (self.data - self.means[k])) / self.z[:,k].sum()) + (1e-6 * np.eye(self.d))
        cov = cov - Gk
        cov = self.get_closest_pos_def(cov)
        self.cov[k] = cov # cov: new d*d covariance matrix for component k --> equation 15
        # assert (not (not np.allclose(cov, cov.T) or np.any(np.linalg.eigvalsh(cov) <= 0)))
        assert np.allclose(self.cov[k],self.cov[k].T)
        print('difference with transpose')
        print(cov-cov.T)


    def h(self,data,k):
        """gamma weights for the bounded version of SMM --> equation 17
        returns a np array of n gamma weights, where n  = data.shape[0]"""
        ret = np.empty(shape=data.shape[0])
        for i in range(data.shape[0]):
            numer = self.degs[k] + self.d
            mah = distance.mahalanobis(data[i,:], self.means[k], self.cov[k])
            denom = self.degs[k] + mah
            ret[i] = numer / denom
        return ret

    def h_indiv(self,x,k):
        """gamma weights for the bounded version of SMM --> equation 17
        returns a np array of n gamma weights, where n  = data.shape[0]"""
        numer = self.degs[k] + self.d
        mah = distance.mahalanobis(x, self.means[k], self.cov[k])
        denom = self.degs[k] + mah
        ret = numer / denom
        return ret

    @staticmethod
    def get_closest_pos_def(x):
        '''gets the closest positive definite matrix to the input matrix x
        this function will be used to adjust the covariance matrix after its update'''
        y = (x + x.T) / 2
        eig, v = np.linalg.eig(y)
        d = np.diag(eig)
        d[d < 0] = 0
        # ret = np.matmul(np.matmul(v, d), np.linalg.inv(v))
        dim = x.shape[0]
        ret = np.matmul(np.matmul(v, d), v.T)
        ret = ret + np.identity(dim) * 10e-8
        return ret