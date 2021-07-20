import torch
import tarfile
import pickle
import pandas
from pathlib import Path
import numpy as np
from numpy import asarray
import os
import re
import json
import pandas as pd
import sys
from tslearn.clustering import TimeSeriesKMeans, KShape
import torch
import tarfile
from sklearn.metrics import log_loss
import pickle
import pandas
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
from shutil import copyfile
import os
import re
import pandas as pd
import sys
from numpy import asarray
from numpy import savetxt
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
def purity(learned_ids, gt_ids):
    """
    Args:
    - learned_ids - 1-D tensor of labels obtained from model
    - gt_ids - 1-D tensor of ground truth labels
    """
    print(len(learned_ids), len(gt_ids), "Длины")
    assert len(learned_ids) == len(gt_ids)
    pur = 0
    ks = torch.unique(learned_ids)
    js = torch.unique(gt_ids)
    for k in ks:
        inters = []
        for j in js:
            inters.append(((learned_ids == k) * (gt_ids == j)).sum().item())
        pur += 1./len(learned_ids) * max(inters)

    return pur
def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
import tslearn
from tslearn.utils import to_time_series_dataset
sys.path.append("..")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--save_dir', type=str, required=True, help='dir holding sequences as separate files')
    args = parser.parse_args()
    return args
def K_shape(args):
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    (str, Path) - directory containing sequences
        save_dir

    Returns:
             
    """
    data_dir = args.data_dir
    #K = args.K
    os.mkdir(args.save_dir)
    gt = pd.read_csv(Path(data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
    K = len(np.unique(gt))
    seq_nmb = len(gt)

    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    info_score = np.zeros((K+1, K+1))
    time_start = time.clock()
    ext = 'csv'
    data = []
    ts = []
    leng = 0
    events_arr0 = []
    for file in sorted(os.listdir(data_dir), key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0): 
        if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
            f = pd.read_csv(Path(data_dir, file))
            leng = max(leng, len(f[event_col]))
            for event in f[event_col].to_numpy():
                events_arr0.append(event)
    events_arr=np.unique(events_arr0)
    events = len(events_arr)
    for file in sorted(os.listdir(data_dir), key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0): 
        if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
          f = pd.read_csv(Path(data_dir, file))
          if f[time_col].to_numpy()[-1] < 0:
                 continue
          for event_type in range(len(events_arr)):
              d = np.zeros(leng)#(len(f[event_col]))
              print("d shape", d.shape)
              dat = f[f[event_col] == events_arr[event_type]][time_col].to_numpy()
              for k in range (len(dat)):
                d[k]= dat[k]
              #print("Appendingevent \n", event_type)
              data.append(d)

          ts.append(asarray(data)) 
    print('DS comleted')
    Ts = to_time_series_dataset((asarray(ts)))
    Ts1 = np.zeros((len(Ts), events, Ts.shape[2]))
    for i in range(len(Ts)):
        Ts1[i] = Ts[i][:events]
    model = KShape(n_clusters=K, max_iter=5)
    labels1 = model.fit_predict(Ts1)
    labels = torch.LongTensor(labels1)
    gt_ids = torch.LongTensor(gt)
    times = time.clock() - time_start
    if gt_ids is not None:
        pur_val_mean =purity(labels, gt_ids) 
    print(f'Purity kshap: {pur_val_mean:.4f}')
        #print(f'Normalized mutual info score: {info_score}')
    print(f'Mean run time kshape: {times}')
    if (gt_ids is not None):
        metrics = {
            "Purity": f'{pur_val_mean:.4f}' ,
            "Mean run time": f'{times:.4f}',
            "Normalized mutual info score:": f'{info_score}',
            #"Predictive log likelihood:":f'{log_loss(gt_ids, labels):.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.save_dir, "k_shape_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        metrics = {
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.save_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
    time_start = time.clock()
    model = TimeSeriesKMeans(n_clusters=K, metric="softdtw", max_iter=5)
    labels1 = model.fit_predict(Ts1)
    labels_kmean = torch.LongTensor(labels1)
    gt = pd.read_csv(Path(data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
    gt_ids = torch.LongTensor(gt)
    times = time.clock() - time_start
    if gt_ids is not None:
        pur_val_mean =purity(labels_kmean, gt_ids) 
        print(f'Purity kmeans: {pur_val_mean:.4f}')
        print(f'time kmeans: {times}')
    #print(f'Mean run time: {time_mean:.4f}+-{time_std:.4f}')
    if (gt_ids is not None):
        metrics = {
            "Purity": f'{pur_val_mean:.4f}' ,
            "Mean run time": f'{times:.4f}',
            "Normalized mutual info score:": f'{info_score}',
            #"Predictive log likelihood:":f'{log_loss(gt_ids, labels):.4f}',
            "Predicted labels":f'{labels_kmean}'
        }
        with open(Path(args.save_dir, "k_means_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    res_df = pd.read_csv(os.path.join(args.data_dir, "clusters.csv"))
        #res_df["time"] = round(times,4)
    res_df["seqlength"] = 0
    for index, row in res_df.iterrows():
        seq_df = pd.read_csv(os.path.join(args.data_dir, str(index + 1) + ".csv"))
        res_df.at[index, "seqlength"] = len(seq_df)
    res_df["kshape_cluster"] = labels.cpu().numpy().tolist()
    res_df["kmeans_cluster"] = labels_kmean.cpu().numpy().tolist()
    savepath = os.path.join(args.save_dir, "inferredclusters.csv")
    res_df.drop(
        res_df.columns[res_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    res_df.to_csv(savepath)
    return ts
args = parse_arguments()
print(args)
K_shape(args)