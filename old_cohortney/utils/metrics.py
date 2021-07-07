"""
    This file contains important metrics
"""
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


def log_likelihood_single(partitions, lambdas, dts):
    """
        input:
               partitions - torch.Tensor, size = (batch size, sequence length, number of classes + 1), input data
               lambdas - torch.Tensor, size = (batch size, sequence length, number of classes), model output
               dts - torch.Tensor, size = (batch size), delta times for each sequence

        output:
               log likelihood - torch.Tensor, size = (1)
    """
    tmp1 = lambdas * dts[:, None, None]
    p = partitions[:, :, 1:]
    return torch.sum(tmp1) - torch.sum(p * torch.log(tmp1))


def info_score(learned_ids, gt_ids, K):
    """
        input:
               learned_ids - torch.Tensor, labels obtained from model
               gt_ids - torch.Tensor, ground truth labels
               K - number of clusters

        output:
                   info_score - np.array: nmb_clusters+1 x nmb_clusters+1), where [i,j] elementis mutual info scorebetween i and j clusters
    """
    assert len(learned_ids) == len(gt_ids)
    info_score = np.zeros((K + 1, K + 1))
    for k in range(1, K + 1):
        info_score[k, 0] = k - 1
        for j in range(1, K + 1):
            info_score[0, j] = j - 1
            ind = np.concatenate([np.argwhere(gt_ids == j - 1), np.argwhere(gt_ids == k - 1)], axis=1)[0]
            learned_idsl = learned_ids.tolist()
            gt_idsl = gt_ids.tolist()
            info_score[k, j] += normalized_mutual_info_score([learned_idsl[i] for i in ind],
                                                             [gt_idsl[i] for i in ind]) 
    return info_score


def purity(learned_ids, gt_ids):
    """
        input:
               learned_ids - torch.Tensor, labels obtained from model
               gt_ids - torch.Tensor, ground truth labels

        output:
               purity - float, purity of the model
    """
    assert len(learned_ids) == len(gt_ids)
    pur = 0
    ks = torch.unique(learned_ids)
    js = torch.unique(gt_ids)
    for k in ks:
        inters = []
        for j in js:
            inters.append(((learned_ids == k) * (gt_ids == j)).sum().item())
        pur += 1. / len(learned_ids) * max(inters)

    return pur
