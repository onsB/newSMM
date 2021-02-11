from BSMM_New import *
# import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return posterior.argmax(axis=1)

def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    n_comp = np.unique(labels).shape[0]
    bsmm = BSMM(X=data, n_components=n_comp)
    bsmm._init()
    # print('prior')
    # print(bsmm.prior)
    # print('sums up to %d' % bsmm.prior.sum())
    # print('up:')
    # print(bsmm.up)
    # print('low: ')
    # print(bsmm.low)
    # print('means:')
    # print(bsmm.means)
    # print('cov:')
    # print(bsmm.cov)

    bsmm.fit(tol= 1e-10)

    # print('after modifications')
    # print('prior')
    # print(bsmm.prior)
    # print('sums up to %d' % bsmm.prior.sum())
    # print('up:')
    # print(bsmm.up)
    # print('low: ')
    # print(bsmm.low)
    # print('means:')
    # print(bsmm.means)
    # print('cov:')
    # print(bsmm.cov)
    #h = bsmm.h(data,1)
    #print('h shape: ', h.shape)
    #print(h[:5])
    # z, resp = bsmm.E_step()
    # print('z matrix')
    # print(z[:3])
    # print('resp matrix')
    # print(resp[:3])
    print('*** Results ***')
    print_performance(bsmm.z,labels)
    print('components: ',bsmm.n_components)

    #sns.heatmap(bsmm.cov[1], xticklabels=False, yticklabels=False)
    #plt.show()

    # heart = pd.read_csv('heart_dataset/heart_reduced.csv')
    # data2 = heart[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].to_numpy()
    # scaler = StandardScaler()
    # data2 = scaler.fit_transform(data2)
    # labels2 = heart['target'].to_numpy()
    #
    # n_comp2 = np.unique(labels2).shape[0]
    # bsmm2 = BSMM(X=data2, n_components=n_comp2)
    # bsmm2._init()
    # print('**second dataset**')
    # print('prior ')
    # print(bsmm2.prior)
    # print('sums up to %d'% bsmm2.prior.sum())
    # print('up:')
    # print(bsmm2.up)
    # print('low: ')
    # print(bsmm2.low)
    #print('means:')
    #print(bsmm2.means)
    #print('cov:')
    #print(bsmm2.cov)
    # bsmm2.fit()
    # print('after modifications')
    #print('prior')
    #print(bsmm2.prior)
    # print('sums up to %d' % bsmm2.prior.sum())
    # print('up:')
    # print(bsmm2.up)
    # print('low: ')
    # print(bsmm2.low)
    # print('means:')
    # print(bsmm2.means)
    # print('cov:')
    # print(bsmm2.cov)
    # print('*** Results ***')
    # print_performance(bsmm2.z,labels2)
    # print('components: ',bsmm2.n_components)


if __name__ == "__main__":
    main()
