import t_pdf
import kmeans_init
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
n,d = data.shape
labels = iris.target
K = np.unique(labels).shape[0]
params = kmeans_init.init_params(data, K)

means = params['means']
prior = params['prior']
cov = params['cov_mat']
degs = params['degs']
low = params['lower_bound']
up = params['upper_bound']

for k in range(K):
    for i in range(data.shape[0]):
        tmp = t_pdf.bounded_multivariate_t_pdf(data[i,:], means[k], cov[k], degs[k], up[k], low[k])
        if tmp <= 0:
            print(data[i])
            print('i: %d'%i)
            print('k: %d'%k)
            print('pdf: ',tmp)

means2 = np.array(np.random.random((K,d)) + np.mean(data))
cov2 = np.array([np.asmatrix(np.identity(d)) for i in range(K)])
prior2 = np.ones(K)/K
degs2 = np.random.randint(n, size=K)
up2 = np.zeros((K, d)) + np.max(data, axis =0)
low2 = np.zeros((K, d)) + np.min(data, axis=0)

for k in range(K):
    for i in range(data.shape[0]):
        tmp = t_pdf.bounded_multivariate_t_pdf(data[i,:], means2[k], cov2[k], degs2[k], up2[k], low2[k])
        if tmp <= 0:
            print(data[i])
            print('i: %d'%i)
            print('k: %d'%k)
            print('pdf: ',tmp)