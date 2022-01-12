import argparse
import collections
import glob
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def purity(gt_ids: torch.Tensor, learned_ids: torch.Tensor) -> float:
    """
    :arg:
           learned_ids - torch.Tensor, labels obtained from model
           gt_ids - torch.Tensor, ground truth labels
    :return:
           purity - float, purity of the model
    """
    gt_ids = torch.Tensor(gt_ids)
    learned_ids = torch.Tensor(learned_ids)
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


def metrics_map(metrics_name: str):
    """
    Maps metrics name from results dictionary to metrics function
    """
    metrics_dict = {
        "purities": purity,
        "adj_mut_info_score": adjusted_mutual_info_score,
        "adj_rand_score": adjusted_rand_score,
        "v_meas_score": v_measure_score,
        "f_m_score": fowlkes_mallows_score,
    }

    return metrics_dict[metrics_name]


def cohortney_tsfresh_stats(dataset: str, methods_list: List[str]):
    """
    Function to obtain summary statistics of Cohortney/Tsfresh on dataset
    """
    res_dict = collections.defaultdict(dict)
    for method in methods_list:
        exp_folder = os.path.join("../experiments", method, dataset)
        experiments = glob.glob(exp_folder + "/exp_*")
        mtrcs = [
            "purities",
            "adj_mut_info_score",
            "adj_rand_score",
            "v_meas_score",
            "f_m_score",
        ]

        res_dict[method]["time"] = 0
        res_dict[method]["train_time"] = 0
        for metr in mtrcs:
            res_dict[method][metr] = []

        # iterating over experiments resuls
        n_runs = 0
        for exp in experiments:

            clusters = os.path.join(exp, "inferredclusters.csv")
            if os.path.exists(clusters):
                df = pd.read_csv(clusters)
                true_labels = df["cluster_id"].to_list()
                if method == "cohortney":
                    res_dict[method]["time"] += df["time"][0]
                    pred_labels = df["coh_cluster"].to_list()
                elif method == "cae":
                    pred_labels = df["cluster_cohortney"].to_list()
                elif method == "dmhp":
                    pred_labels = df["zhu_cluster"].to_list()
                else:
                    pred_labels = df["cluster_" + method.split("_")[-1]].to_list()
                for metrics_name in mtrcs:
                    single_score = metrics_map(metrics_name)(true_labels, pred_labels)
                    res_dict[method][metrics_name].append(single_score)

            n_runs += 1
            # training time
            res_file = os.path.join(exp, "results.pkl")
            if os.path.exists(res_file):
                # n_runs += 1
                with open(res_file, "rb") as f:
                    res_list = pickle.load(f)
                res_dict["cohortney"]["train_time"] += res_list[-1][4]

        res_dict[method]["n_runs"] = n_runs
        res_dict[method]["n_clusters"] = len(np.unique(np.array(true_labels)))

    return res_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="exp_K2_C5")
    args = parser.parse_args()
    methods = [
        "cohortney",
        "dmhp",
        "cae",
        "thp",
        "kmeans_tslearn",
        "kshape_tslearn",
        "kmeans_tsfresh",
        "gmm_tsfresh",
    ]
    res_dict = cohortney_tsfresh_stats(args.dataset, methods)
    print("dataset", args.dataset)
    print("number of runs", res_dict["gmm_tsfresh"]["n_runs"])
    for m in methods:
        print(m)
        # print("mean alg time", res_dict[m]["time"] / res_dict[m]["n_runs"])
        print("mean purity", np.mean(np.array(res_dict[m]["purities"])))
        print("stdev purity", np.std(np.array(res_dict[m]["purities"])))
