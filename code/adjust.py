import numpy as np

def find_closest_pos_def(x):
    y = (x + x.T)/2
    eig, v = np.linalg.eig(y)
    d = np.diag(eig)
    d[d<0] = 0
    ret = np.matmul(np.matmul(v, d), np.linalg.inv(v))

    dim = x.shape[0]
    ret = ret + np.identity(dim) * 10e-8

    ret2 = np.matmul(np.matmul(v, d), v.T)
    ret2 = ret2 + np.identity(dim) * 10e-8

    return ret, ret2

def test(m):
    return not (not np.allclose(m, m.T) or np.any(np.linalg.eigvalsh(m) <= 0))