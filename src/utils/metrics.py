"""
    This file contains metrics for even sequence clustering evaluation
"""

import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score


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


def info_score(learned_ids: torch.Tensor, gt_ids: torch.Tensor, K: int) -> np.array:
    """
    :arg:
           learned_ids - torch.Tensor, labels obtained from model
           gt_ids - torch.Tensor, ground truth labels
           K - number of clusters

    :return:
         info_score - np.array: n_clusters+1 x n_clusters+1,
         where [i,j] element is mutual info score between i and j clusters
    """
    assert len(learned_ids) == len(gt_ids)
    info_score = np.zeros((K + 1, K + 1))
    for k in range(1, K + 1):
        info_score[k, 0] = k - 1
        for j in range(1, K + 1):
            info_score[0, j] = j - 1
            ind = np.concatenate(
                [np.argwhere(gt_ids == j - 1), np.argwhere(gt_ids == k - 1)], axis=1
            )[0]
            learned_idsl = learned_ids.tolist()
            gt_idsl = gt_ids.tolist()
            info_score[k, j] += normalized_mutual_info_score(
                [learned_idsl[i] for i in ind], [gt_idsl[i] for i in ind]
            )
    return info_score


def purity(gt_ids: torch.Tensor, learned_ids: torch.Tensor) -> float:
    """
    :arg:
           learned_ids - torch.Tensor, labels obtained from model
           gt_ids - torch.Tensor, ground truth labels

    :return:
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
        pur += 1.0 / len(learned_ids) * max(inters)

    return pur


def consistency(trials_labels):
    """
    Args:
    - trials_labels - array-like sequence of 1-D tensors. Each tensor is a sequence of labels
    """
    J = len(trials_labels)
    values = torch.zeros(J)

    for trial_id, labels in enumerate(trials_labels):
        ks = torch.unique(labels)
        sz_M = 0  # number of pairs within same cluster
        for k in ks:
            mask = labels == k
            sz = mask.sum()
            s = sz * (sz - 1.0) / 2.0
            sz_M += s

        for trial_id2, labels2 in enumerate(trials_labels):
            if trial_id == trial_id2:
                continue

            for k in ks:
                mask = labels == k
                s2 = 0
                for k2 in labels2[mask].unique():
                    sz = (
                        labels2[mask] == k2
                    ).sum()  # same cluster within j trial, same cluster within j' trial
                    s2 += sz * (sz - 1.0) / 2.0
                # values[trial_id] += (sz_M - s2) / ((J-1) * sz_M)
                values[trial_id] += s2
        values[trial_id] /= (J - 1) * sz_M

    return torch.min(values)
