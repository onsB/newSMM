import numpy as np

def fisher_info(x, params):
    n,d = x.shape
    m = params['prior'].shape
    f_weights = np.power(n, m-1)
    prod = 1
    for w in params['prior']:
        prod = prod * w
    f_weights = f_weights / prod
