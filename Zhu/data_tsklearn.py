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
def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
from HP import PointProcessStorage, DirichletMixtureModel, EM_clustering, tune_basis_fn
from metrics import consistency, purity
import tslearn
from tslearn.utils import to_time_series_dataset
sys.path.append("..")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--K', type=int, required=True, help='clusters')
    parser.add_argument('--events', type=int, required=True, help='events')
    args = parser.parse_args()
    return args
def load_data(args):
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    (str, Path) - directory containing sequences


    Returns:
        ss          (List(torch.Tensor))    - list of torch.Tensor containing sequences. Each tensor has shape (L, 2) and represents event sequence 
                                                as sequence of pairs (t, c). t - time, c - event type.
        Ts          (torch.Tensor)          - tensor of right edges T_n of interavls (0, T_n) in which point processes realizations lie.
        class2idx   (Dict)                  - dict of event types and their indexes
        user_list   (List(Dict))            - representation of sequences siutable for Cohortny
             
    """
    data_dir = args.data_dir
    K = args.K
    events = args.events
    with open(Path(data_dir, 'info.json')) as info:
        info = json.load(info)
    classes = info['classes']
    seq_nmb = info['seq_nmb']
    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    info_score = np.zeros((K+1, K+1))
    time_start = time.clock()

    class2idx = {clas: idx for idx, clas in enumerate(classes)}
    data = []
    ts = []
    print('events', class2idx.values())
    leng = 0
    for i in range(1, seq_nmb+1):
        f = pd.read_csv(Path(data_dir, f'{i}.csv'))
        leng = max(leng, len(f[event_col]))
    for i in range(1, seq_nmb+1):
      print (i)
      f = pd.read_csv(Path(data_dir, f'{i}.csv'))
      if f[time_col].to_numpy()[-1] < 0:
             continue

      for event_type in class2idx.values():
          d = np.zeros(leng)#(len(f[event_col]))
          print("d shape", d.shape)
          dat = f[f[event_col] == event_type][time_col].to_numpy()
          for k in range (len(dat)):
            d[k]= dat[k]
          #print("Appendingevent \n", event_type)
          data.append(d)
          
      ts.append(asarray(data)) 
    print('DS comleted')
    print(type(asarray(ts)[1]), asarray(ts)[1].shape)
    Ts = to_time_series_dataset((asarray(ts)))
    Ts1 = np.zeros((len(Ts), events, Ts.shape[2]))
    for i in range(len(Ts)):
        Ts1[i] = Ts[i][:events]
    print(Ts1[:2])
    model = TimeSeriesKMeans(n_clusters=K, metric="softdtw", max_iter=5)
    labels1 = model.fit_predict(Ts1)
    
    print('Model', labels1[:3])
    labels = torch.LongTensor(labels1)
    print('labels',labels)
    gt = pd.read_csv(Path(data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
    print('gt', gt)
    gt_ids = torch.LongTensor(gt)
    print('gt', len(gt_ids))
    times = time.clock() - time_start
    if gt_ids is not None:
        pur_val_mean =purity(labels, gt_ids) 
        #print(f'Purity: {pur_val_mean:.4f}+-{pur_val_std:.4f}')
        #print(f'Normalized mutual info score: {info_score}')
    #print(f'Mean run time: {time_mean:.4f}+-{time_std:.4f}')
    if (gt_ids is not None):
        metrics = {
            "Purity": f'{pur_val_mean:.4f}' ,
            "Mean run time": f'{times:.4f}',
            "Normalized mutual info score:": f'{info_score}',
            #"Predictive log likelihood:":f'{log_loss(gt_ids, labels):.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.data_dir, "k_means_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        metrics = {
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
    return ts
args = parse_arguments()
print(args)
load_data(args)