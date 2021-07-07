import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


plt.style.use("science")


# Figure 2
dataset_names = ["new_sin_K5_C5", "K3_C5", "Age"]
coh_ll = []
for dataset in dataset_names:
    exper_path = os.path.join("../experiments", dataset)
    # should calculate true ll of dgp of dataset
    true_ll = 0
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
    mean_ll = np.mean(total_ll, axis=1)
    mean_ll = mean_ll - true_ll
    coh_ll.append(mean_ll)

data_coh = coh_ll
data_zhu = [[3*10**6], [3*10**5], [2*10**6]]
dataset_names = ["sin\_K5\_C5", "K3\_C5", "Age"]

plt.figure()
bpl = plt.boxplot(data_coh, positions=np.array(range(len(data_coh)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_zhu, positions=np.array(range(len(data_zhu)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Cohortney')
plt.plot([], c='#2C7BB6', label='Zhu')
plt.legend(fontsize='x-small')
# label x-ticks
plt.xticks(range(0, len(dataset_names) * 2, 2), dataset_names)
plt.xlim(-2, len(dataset_names)*2)
# draw gray strips
plt.axvspan(-0.9, 0.9, facecolor='gray', alpha=0.5)
plt.axvspan(3.1, 4.9, facecolor='gray', alpha=0.5)
plt.savefig("fig2.pdf", dpi=400, bbox_inches="tight")
