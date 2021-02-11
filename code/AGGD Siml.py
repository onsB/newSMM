import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.special import gamma
from sklearn import datasets
from sklearn.cluster import KMeans

class MCMC:

    def __init__(self):
        self.chain = []
        self.target = None

    # def draw(self, x, bins=50):
    #     #     # x = pd.Series(self.chain.tolist(), name="Sample from MCMC")
    #     #     sns.distplot(self.chain, bins)
    #     #     sns.lineplot(x, self.target(x, is_ln=False), lw=2)
    #     #     plt.show()

    def sample(self, x, target, iterations=int(1e5)):
        self.target = target
        return self.mh_sampler(x, target, self.gaussian_proposal_log_pdf, iterations=iterations)

    def mh_sampler(self, x, target_pdf, prop_fn, prop_fn_kwargs={}, iterations=int(1e5), is_ln_prob=True):
        """Simple metropolis hastings sampler.

        :param x: Initial array of parameters.
        :param target_pdf: Function to compute log-posterior.
        :param prop_fn: Function to perform jumps.
        :param prop_fn_kwargs: Keyword arguments for proposal function
        :param iterations: Number of iterations to run sampler. Default=100000

        :returns:
            (chain, acceptance, lnprob) tuple of parameter chain , acceptance rate
            and log-posterior chain.
        """

        # number of dimensions
        ndim = x.shape[1]

        # initialize chain, acceptance rate and lnprob
        chain = np.zeros((iterations, ndim))
        lnprob = np.zeros((iterations, ndim))
        accept_rate = np.zeros(iterations)

        chain = []
        lnprob = []
        # first samples
        chain.append(x)
        lnprob0 = target_pdf(x)
        lnprob.append(lnprob0)

        # start loop
        naccept = 0
        accept_rate = []
        for ii in range(1, iterations):

            # propose
            x_star, factor = prop_fn(x, **prop_fn_kwargs)

            # draw random uniform number
            u = np.random.uniform(0, 1, x.shape)

            # compute hastings ratio
            lnprob_star = target_pdf(x_star)

            H = (np.exp(lnprob_star - lnprob0) * factor) if is_ln_prob else ((lnprob_star / lnprob0) * factor)

            # accept/reject step (update acceptance counter)
            if (u < H).any():
                temp_x = x
                temp_lnprob0 = lnprob0
                x = np.where(u < H, x_star, temp_x)
                lnprob0 = np.where(u < H, lnprob_star, temp_lnprob0)
                naccept += 1

            # update chain
            chain.append(x)
            lnprob.append(lnprob0)
            accept_rate.append(naccept / ii)

        self.chain = chain
        return chain, accept_rate, lnprob

    def gaussian_proposal_log_pdf(self, x, sigma=1):
        """
        Gaussian proposal distribution.

        Draw new parameters from Gaussian distribution with
        mean at current position and standard deviation sigma.

        Since the mean is the current position and the standard
        deviation is fixed. This proposal is symmetric so the ratio
        of proposal densities is 1.

        :param x: Parameter array
        :param sigma:
            Standard deviation of Gaussian distribution. Can be scalar
            or vector of length(x)

        :returns: (new parameters, ratio of proposal densities)
        """
        # import tensorflow_probability as tfp

        # Draw x_star
        x_star = x + np.random.randn(x.shape[0], x.shape[1]) * sigma
        # x_star = np.log(scipy.stats.norm(loc=0, scale=sigma).pdf(x))

        # x_star = tfp.distributions.Normal(loc=[0.], scale=[1.]).log_prob(x)
        # proposal ratio factor is 1 since jump is symmetric
        qxx = 1

        return x_star, qxx

def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n).toarray()

class AsymmetricGeneralizedGaussian(object):

    def __init__(self, params, k):
        self.mean = params['means'][k]
        self.sigma_l = params['standard deviation l'][k]
        self.sigma_r = params['standard deviation r'][k]
        self.shape = params['shape'][k]
        self.coefficient = np.divide(np.multiply(self.shape, self.custom1()), np.multiply((self.sigma_l+self.sigma_r), gamma(np.divide(1, self.shape))))

    def custom1(self):
        ret = gamma(np.divide(3, self.shape))/gamma(np.divide(1,self.shape))
        return np.power(ret, 0.5)
    def alpha_custom(self):
        ret = gamma(np.divide(3,self.shape))/gamma( np.divide(1,self.shape))
        return np.power(ret, np.divide(self.shape, 2))
    def custom2(self,x,flag):
        if flag.all():
            ret = np.divide((self.mean-x), self.sigma_l)
        else:
            ret = np.divide((x-self.mean), self.sigma_r)
        return np.power(ret,self.shape)
    def _f(self, new_x, sigma):
        if sigma == self.sigma_l:
            flag = True
        else:
            flag = False

        return np.multiply(self.coefficient, np.exp(np.multiply(-self.alpha_custom(), self.custom2(new_x, self.custom2(new_x, flag)))))

    def pdf(self, X):

        return np.where(X - self.mean < 0, self._f(X, self.sigma_l),
                            self._f(X, self.sigma_r))

    def ln_pdf(self, X):
        def f(x, sigma):
            return np.log(self.coefficient) - (np.multiply(self.alpha_custom(), self.custom2(x, sigma == self.sigma_l)) )


        return np.where((X - self.mean) < 0, f(X, self.sigma_l), f(X, self.sigma_r))

def sim_aggmm(params,k,n1):
    def sim_target(x, is_ln=True):


        agm = AsymmetricGeneralizedGaussian(params, k)
        if is_ln:
            return agm.ln_pdf(x)
        else:
            return agm.pdf(x)


    mcmc = MCMC()
    ret = []
    # for i in range(n1):
    x = np.random.randn(n1, params['means'].shape[1])
    chain, ar, log_prob = mcmc.sample(x, sim_target, int(2e3))
    ret = chain[-1]

    return ret


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

    dim = data.shape[1]
    lam = []
    for k in range(num_clusters):
        lam.append(2 * np.ones(dim))

    right_bound = []
    left_bound = []
    for i in unsorted_labels:
        right_bound.append(np.amin(data[cluster_assignment == i], axis=0))
        left_bound.append(np.amax(data[cluster_assignment == i], axis=0))
    Bounded_Support = {'right bound': np.array(right_bound), 'left bound': np.array(left_bound)}

    params = {'prior': np.array(weights), 'means': np.array(means), 'standard deviation l': np.array(covs), 'standard deviation r': np.array(covs), 'shape': np.array(lam)}
    useless_params = {'prior': np.array(weights), 'means': np.array(means), 'standard deviation': np.array(covs),
              'shape': np.array(lam)}
    return params, Bounded_Support, useless_params

iris = datasets.load_iris()
Breast_Cancer = datasets.load_breast_cancer()
# 1 IRIS
# Observations = iris.data
# Targets = iris.target
# 2 BREAST CANCER
Observations = Breast_Cancer.data
Targets = Breast_Cancer.target
k_groundtruth = np.unique(Targets).shape[0]
sim_parameters, main_bounds, not_important = init_params(Observations, k_groundtruth)
sample_perClass = int(1e3)

def sim_allk(params, samples_k):
    all_k = params['means'].shape[0]
    all_d = params['means'].shape[1]
    ret_data = sim_aggmm(params, 0, samples_k)
    ret_targets = np.zeros((samples_k,1))
    for k in range(1, all_k):
        ret_data = np.concatenate((ret_data, sim_aggmm(params, k, samples_k)), axis=0)
        ret_targets = np.concatenate((ret_targets, np.ones((samples_k, 1)) * k), axis=0)
    return ret_data, ret_targets

result_obs = sim_allk(sim_parameters, sample_perClass)