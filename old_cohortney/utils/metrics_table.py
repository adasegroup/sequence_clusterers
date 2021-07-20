import argparse
import collections
import glob
import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
#from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score #(labels_true, labels_pred, beta=1.0)
from sklearn.metrics.cluster import fowlkes_mallows_score
from metrics import purity

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="K2_C5")
parser.add_argument("--method", type=str, default="cohortney")
args = parser.parse_args()
exp_folder = os.path.join("../experiments", args.dataset)
experiments = glob.glob(exp_folder + "/exp_*")


res_dict = {}
res_dict["time"] = 0
res_dict["purities"] = []
res_dict["adjusted_mutual_info_score"] = []
res_dict["adjusted_rand_score"] = []
res_dict["fowlkes_mallows_score"] = []
res_dict["v_measure_score"] = []
# iterating over experiments resuls
n_runs = 0
for exp in experiments:
    time_0 = time.clock() 
    clusters = os.path.join(exp, "compareclusters.csv")
    if os.path.exists(clusters):
        n_runs += 1
        df = pd.read_csv(clusters)
        labels_true = df["cluster_id"].to_list()
        labels_pred = df["coh_cluster"].to_list()
        res_dict["time"] = res_dict["time"] - df["time"][0]
        pur = purity(torch.FloatTensor(labels_true), torch.FloatTensor(labels_pred))
        amis = adjusted_mutual_info_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        fms = fowlkes_mallows_score(labels_true, labels_pred)
        v = v_measure_score(labels_true, labels_pred, beta=1.0)
       
        time_exp = time.clock() - time_0
        res = {}
        res["purities"] = pur
        time_exp = time.clock() - time_0
        res["time"] = time_exp
        res["adjusted_mutual_info_score"] = amis
        res["adjusted_rand_score"] = ari
        res["fowlkes_mallows_score"] = fms
        res["v_measure_score"] = v
        res_dict["purities"].append(pur)
      
        res_dict["adjusted_mutual_info_score"].append(amis)
        res_dict["adjusted_rand_score"].append(ari)
        res_dict["fowlkes_mallows_score"].append(fms)
        res_dict["v_measure_score"].append(v)
        res_file = open(os.path.join(exp, "metrics.csv"), "w")
        writer = csv.writer(res_file)
        for key, value in res.items():
            writer.writerow([key, value])
        res_file.close()
res = {}
res["time"] = res_dict["time"]/n_runs
res["adjusted_mutual_info_score"] = [np.mean(np.array(res_dict["adjusted_mutual_info_score"])), np.std(np.array(res_dict["adjusted_mutual_info_score"]))]
res["adjusted_rand_score"] = [np.mean(np.array(res_dict["adjusted_rand_score"])), np.std(np.array(res_dict["adjusted_rand_score"]))]
res["fowlkes_mallows_score"] = [np.mean(np.array(res_dict["fowlkes_mallows_score"])), np.std(np.array(res_dict["fowlkes_mallows_score"]))]
res["v_measure_score"] = [np.mean(np.array(res_dict["v_measure_score"])), np.std(np.array(res_dict["v_measure_score"]))]
res["purities"] = [np.mean(np.array(res_dict["purities"])), np.std(np.array(res_dict["purities"]))]

res_file = open(os.path.join(exp_folder, "metrics.csv"), "w")
writer = csv.writer(res_file)
for key, value in res.items():
    writer.writerow([key, value])
res_file.close()

print("dataset", args.dataset)
print("number of runs", n_runs)
print("number of clusters", len(np.unique(np.array(labels_true))))
print("mean alg time", res_dict["time"] / n_runs)
print("mean purity", np.mean(np.array(res_dict["purities"])))
print("stdev purity", np.std(np.array(res_dict["purities"])))