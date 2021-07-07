from utils.metrics import purity
import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="K2_C5")
args = parser.parse_args()
exp_folder = os.path.join("experiments", args.dataset) 
experiments = glob.glob(exp_folder+"/exp_*")


methods = ["cohortney", "kmeans", "ap", "gmm"]
#res_dict = dict.fromkeys(methods)
res_dict = collections.defaultdict(dict)
for m in methods:

    res_dict[m]["time"] = 0
    res_dict[m]["purities"] = []


# iterating over experiments resuls
n_runs = 0
for exp in experiments:
    clusters = os.path.join(exp, "compareclusters.csv")
    if os.path.exists(clusters):
        n_runs += 1
        df = pd.read_csv(clusters)
        true_labels = df["cluster_id"].to_list()
        # cohortney
        labels = df["coh_cluster"].to_list()
        res_dict["cohortney"]["time"] += df["time"][0]
        pur = purity(torch.FloatTensor(true_labels),torch.FloatTensor(labels))
        res_dict["cohortney"]["purities"].append(pur)
        # kmeans
        labels = df["kmeans_clusters"].to_list()
        res_dict["kmeans"]["time"] += df["kmeans_time"][0]
        pur = purity(torch.FloatTensor(true_labels),torch.FloatTensor(labels))
        res_dict["kmeans"]["purities"].append(pur) 
        # ap
        labels = df["ap_clusters"].to_list()
        res_dict["ap"]["time"] += df["ap_time"][0]
        pur = purity(torch.FloatTensor(true_labels),torch.FloatTensor(labels))
        res_dict["ap"]["purities"].append(pur)
        # gmm
        labels = df["gmm_clusters"].to_list()
        res_dict["gmm"]["time"] += df["gmm_time"][0]
        pur = purity(torch.FloatTensor(true_labels),torch.FloatTensor(labels))
        res_dict["gmm"]["purities"].append(pur)

print("dataset", args.dataset)
print("number of runs", n_runs)
print("number of clusters", len(np.unique(np.array(true_labels))))
for m in methods:
    print(m)
    print("mean alg time", res_dict[m]["time"]/n_runs)
    print("mean purity", np.mean(np.array(res_dict[m]["purities"])))
    print("stdev purity", np.std(np.array(res_dict[m]["purities"])))
