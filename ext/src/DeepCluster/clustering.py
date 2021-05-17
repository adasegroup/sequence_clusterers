import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.utils.data as data


__all__ = ['Kmeans', 'cluster_assign', 'arrange_clustering']


class ReassignedDataset(data.Dataset):
    """A dataset where the new labels are given in argument.
    Args:
        indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): dataset
    """

    def __init__(self, indexes, pseudolabels, dataset):
        self.dataset = self.make_dataset(indexes, pseudolabels, dataset)

    def make_dataset(self, indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        seqs = []
        for j, idx in enumerate(indexes):
            item = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            seqs.append((item, pseudolabel))
        return seqs

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (seq, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        item, pseudolabel = self.dataset[index]

        return item, pseudolabel

    def __len__(self):
        return len(self.dataset)


# def preprocess_features(npdata, pca=256):
#     """Preprocess an array of features.
#     Args:
#         npdata (np.array N * ndim): features to preprocess
#         pca (int): dim of output
#     Returns:
#         np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
#     """
#     _, ndim = npdata.shape
#     npdata =  npdata.astype('float32')

#     # Apply PCA-whitening with Faiss
#     mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
#     mat.train(npdata)
#     assert mat.is_trained
#     npdata = mat.apply_py(npdata)

#     # L2 normalization
#     row_sums = np.linalg.norm(npdata, axis=1)
#     npdata = npdata / row_sums[:, np.newaxis]

#     return npdata


def cluster_assign(lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        lists (list of list): for each cluster, the list of indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert lists is not None
    pseudolabels = []
    indexes = []
    for cluster, seqs in enumerate(lists):
        indexes.extend(seqs)
        pseudolabels.extend([cluster] * len(seqs))

    return ReassignedDataset(indexes, pseudolabels, dataset)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    kmeans = KMeans(n_clusters=nmb_clusters, init='k-means++', max_iter=20, n_init=1)
    I = kmeans.fit_predict(x)
    loss = kmeans.inertia_

    return [int(n) for n in I], loss


def arrange_clustering(lists):
    pseudolabels = []
    indexes = []
    for cluster, seqs in enumerate(lists):
        indexes.extend(seqs)
        pseudolabels.extend([cluster] * len(seqs))
    indexes = np.argsort(indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        std_sc = StandardScaler()
        xb = std_sc.fit_transform(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss, I
