import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

plt.style.use("science")


# Figure 6
exper_path = "../experiments/new_sin_K5_C5/"
n_runs = 10
for i in range(0, n_runs):
    res_file = os.path.join(exper_path, "exp_" + str(i), "results.pkl")
    with open(res_file, "rb") as f:
        res_list = pickle.load(f)
    # leaving only log ll
    ll = [r[0] for r in res_list]
    if i == 0:
        total_ll = [ll]
    else:
        total_ll.append(ll)

# reshape: from run index to epoch index
total_ll = np.array([np.array(ll) for ll in total_ll])
total_ll = total_ll.T
# mean and std
mean_ll = np.mean(total_ll, axis=1)
std_ll = np.std(total_ll, axis=1)

epochs = list(range(1, len(res_list) + 1))
plt.errorbar(epochs[3:], mean_ll[3:], yerr=std_ll[3:], label="Cohortney")
plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("negative loglikelihood")


plt.savefig("fig6.pdf", dpi=400, bbox_inches="tight")
