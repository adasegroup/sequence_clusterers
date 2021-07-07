import matplotlib.pyplot as plt
import pickle
import os

plt.style.use("science")


# Figure 3
exper_path = "../experiments/new_sin_K5_C5/exp_0"
res_file = os.path.join(exper_path, "results.pkl")

with open(res_file, "rb") as f:
    res_list = pickle.load(f)

epochs = list(range(1, len(res_list) + 1))
# negative log likelihood
ll = [r[0] for r in res_list]
# purities
pur = [r[1] for r in res_list]
# cluster partition
clust_part = [r[2] for r in res_list]
# number of clusters
n_clust = [r[3] for r in res_list]
# time
time_fr_st = [r[4] for r in res_list]

ll = [x / (10**6) for x in ll]
metrics = [ll , pur, clust_part, n_clust, time_fr_st]
fig, axs = plt.subplots(1, len(metrics), constrained_layout=True)
metrics_name = [
    "Negative\nLog Likelihood",
    "Purity",
    "Cluster\npartition",
    "Number\nof clusters",
    "Time",
]

for i, m in enumerate(metrics):

    axs[i].plot(epochs, metrics[i])
    axs[i].set_title(metrics_name[i], fontweight="bold", size=4)
    axs[i].set_xlabel("Epoch", size=4)
    axs[i].tick_params(labelsize=4)
    x0, x1 = axs[i].get_xlim()
    y0, y1 = axs[i].get_ylim()
    axs[i].set_aspect(abs((x1 - x0) / (y1 - y0)))


fig.savefig("fig3.pdf", dpi=400, bbox_inches="tight")
