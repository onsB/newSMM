import numpy as np
from scipy.special import gamma
from scipy.spatial import distance
import math
import data_sampler

def bounded_multivariate_t_pdf(x, mean, cov, deg, up, low, ind):
    m = 5000  # size of the generated data to calculate the normalization constant
    pdf = multi_student_pdf(x, mean, cov, deg)
    norm_const = ind.sum()
    if norm_const == 0 :
        norm_const += 1
    pdf /= norm_const
    if pdf == 0:
        pdf += 1e-6
    return pdf

def multi_student_pdf(x, mean, cov, deg):
    '''computes student's t pdf for a vector x and component k --> equation 4
    '''
    # sig_k = cholesky(cov)
    cov_det = np.power(np.prod(np.diagonal(cov)), 2)
    test = (mean[mean < 0].shape[0] > 0)
    dim = x.shape[0]
    num = gamma((deg + dim) / 2)
    mah = distance.mahalanobis(x, mean, cov)
    product = 1 + mah**2 / deg
    denom = gamma(deg / 2) * np.power(deg * np.pi, dim/2) * math.sqrt(cov_det) * np.power(product, (deg+dim)/2)
    pdf = num/denom
    return pdf

def fill_nans(x):
    assert len(x.shape)==1
    ret = x.copy()
    #colmean = np.nanmean(ret, axis = 0)
    inds = np.where(np.isnan(ret))
    #ret[inds] = np.take(colmean, inds[1])
    ret[inds] = np.nanmean(ret)
    return ret


#delta = xx - yy
#VI = np.linalg.inv(V)
#D = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))