import numpy as np
from sklearn.mixture import BayesianGaussianMixture as BGM
import matplotlib.pyplot as plt

def clusterize_embeddings(data, index_word_map):
    """
    clusters the embeddings using Dirichlet Process Mixture models
    Args:
        data: (np.ndarray) the data matrix
        index_word_map: (dict) (int) index -> (str) word
    Returns:
        labels: (np.ndarray) the cluster labels for the words
    """

    dpgmm = BGM(n_components=20) #chosen at random
    dpgmm.fit(data)
    cluster_ids = dpgmm.predict(data)
    cluster_dict = {}
    for i in range(cluster_ids.shape[0]):
        id_ = cluster_ids[i]
        if id_ not in cluster_dict:
            cluster_dict[id_] = []
        cluster_dict[id_].append(index_word_map[i])

    return cluster_dict


