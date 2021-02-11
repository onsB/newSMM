# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:48:30 2020

@author: onsbo
"""
import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize

def _solve_dof_equation(degs, resp, resp_sum, u, n_dim, tol, n_iter):
    """Solves the equation to calculate the next value of v 
    (degrees of freedom).
    ----------
    degs : array_like, shape (n_components,).
               Degrees of freedoom of ALL the components of the 
               mixture.
    resp : array_like, shape (n_samples, n_components).
        Matrix of responsibilities, each row represents a point 
        and each column represents a component of the mixture.
    resp_sum : array_like, shape (n_samples,).  
            Sum of all the rows of the matrix of 
            responsibilities.
    u : array_like, shape (n_samples, n_components). 
        Matrix of gamma weights, each row represents a point and
        each column represents a component of the mixture.

    n_dim : integer. 
            Number of features of each data point.
    
    Returns
    -------
    new_v_vector : array_like (n_components,).
                   Vector with the updated degrees of freedom for 
                   each component in the mixture.
    """

    n_components = degs.shape[0]

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
