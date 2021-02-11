import numpy as np
from numpy import linalg
from sklearn.cluster import KMeans

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
        # calculate covariance matrix for the ith component
        m_r = data[cluster_assignment == i]  # member rows
        m_r = m_r - m_r.mean(0)
        cov1 = np.matmul(np.transpose(m_r), m_r) / m_r.shape[0]
        testcov1 = not (not np.allclose(cov1, cov1.T) or np.any(linalg.eigvalsh(cov1) <= 0))
        # covs.append(np.sqrt(cov))
        cov = np.cov(m_r.T)
        testcov = not (not np.allclose(cov, cov.T) or np.any(linalg.eigvalsh(cov) <= 0))
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