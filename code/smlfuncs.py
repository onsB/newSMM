import numpy as np
from scipy.special import gamma
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial import distance
import math


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

        # chain = []
        # lnprob = []
        # first samples
        # chain.append(x)
        lnprob0 = target_pdf(x)
        # lnprob.append(lnprob0)

        # start loop
        naccept = 0
        # accept_rate = []
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
            # chain.append(x)
            # lnprob.append(lnprob0)
            # accept_rate.append(naccept / ii)

        self.chain = chain
        # return chain, accept_rate, lnprob
        return x

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


# def diag(array):
#     n = len(array)
#     return spdiags(array, 0, n, n).toarray()

# class AsymmetricGeneralizedGaussian(object):
#
#     def __init__(self, params, k):
#         self.mean = params['means'][k]
#         self.sigma_l = params['standard deviation l'][k]
#         self.sigma_r = params['standard deviation r'][k]
#         self.shape = params['shape'][k]
#         self.coefficient = np.divide(np.multiply(self.shape, self.custom1()),
#                                      np.multiply((self.sigma_l + self.sigma_r), gamma(np.divide(1, self.shape))))
#
#     def custom1(self):
#         ret = gamma(np.divide(3, self.shape)) / gamma(np.divide(1, self.shape))
#         return np.power(ret, 0.5)
#
#     def alpha_custom(self):
#         ret = gamma(np.divide(3, self.shape)) / gamma(np.divide(1, self.shape))
#         return np.power(ret, np.divide(self.shape, 2))
#
#     def custom2(self, x, flag):
#         if flag.all():
#             ret = np.divide((self.mean - x), self.sigma_l)
#         else:
#             ret = np.divide((x - self.mean), self.sigma_r)
#         return np.power(ret, self.shape)
#
#     def _f(self, new_x, sigma):
#
#         return np.multiply(self.coefficient,
#                            np.exp(np.multiply(-self.alpha_custom(), self.custom2(new_x, sigma == self.sigma_l))))
#
#     def pdf(self, X):
#
#         return np.where(X - self.mean < 0, self._f(X, self.sigma_l),
#
#     def ln_pdf(self, X):
#         def f(x, sigma):
#             return np.log(self.coefficient) - (np.multiply(self.alpha_custom(), self.custom2(x, sigma == self.sigma_l)))
#
#         return np.where((X - self.mean) < 0, f(X, self.sigma_l), f(X, self.sigma_r))


class SMM(object):

    def __init__(self, params, k):
        self.mean = params['means'][k]
        self.cov = params['cov_mat'][k]
        self.dof = params['degs'][k]

    def pdf(self, X):
        # sig_k = cholesky(sig_k)
        cov_det = np.power(np.prod(np.diagonal(self.cov)), 2)
        test = (self.cov[self.cov < 0].shape[0] > 0)
        n_samples, dim = X.shape
        pdf = np.zeros(X.shape[0], )
        for i in range(n_samples):
            x_i = X[i]
            num = gamma((self.dof + dim) / 2)
            mah = distance.mahalanobis(x_i, self.mean, self.cov)
            product = 1 + mah / self.dof
            denom = gamma(self.dof / 2) * (self.dof * np.pi) ** (dim / 2) * math.sqrt(cov_det) * product ** (
                        self.dof + dim)
            pdf[i] = np.divide(num, denom)
        return (pdf)

    def ln_pdf(self, X):
        return np.log(self.pdf(X))


def sim_smm(params, k, n1):
    def sim_target(x, is_ln=True):

        smm = SMM(params, k)
        if is_ln:
            return smm.ln_pdf(x)
        else:
            return smm.pdf(x)

    mcmc = MCMC()
    ret = []
    # for i in range(n1):
    # x = np.random.randn(n1, params['means'].shape[1])
    x = np.random.normal(size=(n1, params['means'].shape[1]), loc=params['means'][k])
    chain = mcmc.sample(x, sim_target, int(2e3))
    ret = chain

    return ret


#
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
        # calculate ovariance matrix fr the ith component
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
# Breast_Cancer = datasets.load_breast_cancer()
# # 1 IRIS
Observations = iris.data
Targets = iris.target

# # 2 BREAST CANCER
# Observations = Breast_Cancer.data
# Targets = Breast_Cancer.target


k_groundtruth = np.unique(Targets).shape[0]
sim_parameters = init_params(Observations, k_groundtruth)
sample_perClass = int(1e3)


def sim_allk(params, samples_k):
    all_k = params['means'].shape[0]
    all_d = params['means'].shape[1]
    ret_data = sim_smm(params, 0, samples_k)
    ret_targets = np.zeros((samples_k, 1))
    for k in range(1, all_k):
        ret_data = np.concatenate((ret_data, sim_smm(params, k, samples_k)), axis=0)
        ret_targets = np.concatenate((ret_targets, np.ones((samples_k, 1)) * k), axis=0)
    return ret_data, ret_targets


result_obs = sim_allk(sim_parameters, sample_perClass)
