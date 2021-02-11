# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:20:40 2020

@author: onsbo
"""
from scipy.linalg import cholesky, LinAlgError
import numpy as np

def _cholesky(cv, min_covar):
    """Calculates the lower triangular Cholesky decomposition of a 
    covariance matrix.
    
    Parameters
    ----------
    covar : array_like, shape (n_features, n_features).
            Covariance matrix whose Cholesky decomposition wants to 
            be calculated.
    min_covar : float value.
                Minimum amount that will be added to the covariance 
                matrix in case of trouble, usually 1.e-6.
    Returns
    -------
    cov_chol : array_like, shape (n_features, n_features).
               Lower Cholesky decomposition of a covariance matrix.
    """

    # Sanity check: assert that the covariance matrix is squared
    assert(cv.shape[0] == cv.shape[1])

    # Sanity check: assert that the covariance matrix is symmetric
    if (cv.transpose() - cv).sum() > min_covar:
        print('[SMM._cholesky] Error, covariance matrix not ' \
            + 'symmetric: ' 
            + str(cv)
        )

    n_dim = cv.shape[0]
    try:
        cov_chol = cholesky(cv, lower=True)
    except LinAlgError:
        # The model is most probably stuck in a component with too
        # few observations, we need to reinitialize this components
        cov_chol = cholesky(
            cv + min_covar * np.eye(n_dim), lower=True
        )

    return cov_chol