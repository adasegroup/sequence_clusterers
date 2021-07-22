from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import argparse
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import time
import os
from sklearn.metrics.cluster import normalized_mutual_info_score
def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
from HP import PointProcessStorage, DirichletMixtureModel, EM_clustering, tune_basis_fn
from metrics import consistency, purity
from data_utils import load_data
import tarfile
import pickle
import sys
import json
from sklearn.metrics.cluster import adjusted_mutual_info_score
#from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score #(labels_true, labels_pred, beta=1.0)
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score

sys.path.append("..")

def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
def create_folder(path_to_folder, rewrite=False):
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        if not rewrite:
            return False
        clear_folder(path_to_folder)
        return True
    os.mkdir(path_to_folder)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--nmb_cluster', type=int, default=10, help='number of clusters')
    # hyperparameters for Cohortney
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Tb', type=float, default=7e-6)
    parser.add_argument('--Th', type=float, default=80)
    parser.add_argument('--N', type=int, default=2500)
    parser.add_argument('--n', type=int, default=8, help='n for partition')
    # hyperparameters for training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--save_to', type=str, default="DMHP_Metrics", help='directory for saving metrics')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nruns', type=int, default=7, help='number of trials')
    parser.add_argument('--type', type=str, default=None, help='if it is a')

    parser.add_argument('--result_path', type=str, help='path to save results')
    args = parser.parse_args()
    return args

np.set_printoptions(threshold=10000)
torch.set_printoptions(threshold=10000)

def main(args):
    device = torch.device(args.device)
    ss, Ts, class2idx, gt_ids = load_data(Path("../../pp_clustering31_05/data", args.data_dir))
    N = len(ss)
#     D = 5 # the dimension of Hawkes processes
    w = 1
    
#     basis_fs = [lambda x: torch.exp(- x**2 / (3.*(k+1)**2) ) for k in range(D)]
    exp_folder0 = f"../../Zhu_experiments_1/{args.data_dir}"
    create_folder(exp_folder0 )
    not_tune = True
    eps = 1e5
    basis_fs = tune_basis_fn(ss, eps=eps, not_tune=not_tune)
    D = len(basis_fs) # the dimension of Hawkes processes
    hp = PointProcessStorage(ss, Ts, basis_fs)
    C = len(class2idx)
    K = args.nmb_cluster
    niter = 10
    labels = torch.zeros(args.nruns, len(ss))
    info_score = np.zeros((K+1, K+1))
    nlls = torch.zeros(args.nruns, niter)
    times = np.zeros(args.nruns)

    assigned_labels = []
    results = {}
    for i in range(args.nruns):
        create_folder(exp_folder0 + '/exp_{}'.format(i))
        exp_folder =exp_folder0 + '/exp_{}'.format(i)
        print(f'============= RUN {i+1} ===============')
        time_start = time.clock()
        Sigma = torch.rand(C, C, D, K)
        B = torch.rand(C, K)
        alpha = 1.

        model = DirichletMixtureModel(K, C, D, alpha, B, Sigma)
        print ('model ready')
        EM = EM_clustering(hp, model)
        print ('EM ready')
        r, nll_history, r_history = EM.learn_hp(niter = niter, save = exp_folder, gt_ids = gt_ids, ninner = [2,3,4,5,6,7] + (niter - 6)*[8])
        print ('learn_hp ready')

        labels[i] = r.argmax(-1)
        nlls[i] = torch.FloatTensor(nll_history)
        with open(exp_folder + '/labels.pkl', 'wb') as f:
            pickle.dump(labels[i], f)
        with open(exp_folder + '/args.json', 'w') as f:
            json.dump(vars(args), f)
        torch.save(model, exp_folder + '/last_model.pt')
        print ("preds:", labels[i])
        res_df = pd.read_csv(os.path.join(Path("../../pp_clustering31_05/data", args.data_dir), "clusters.csv"))
 
        #res_df["time"] = round(times,4)
        res_df["seqlength"] = 0
        for index, row in res_df.iterrows():
            seq_df = pd.read_csv(os.path.join(Path("../../pp_clustering31_05/data", args.data_dir), str(index + 1) + ".csv"))
            res_df.at[index, "seqlength"] = len(seq_df)

        res_df["zhu_cluster"] = labels[i].cpu().numpy().tolist()
        savepath = os.path.join(exp_folder, "inferredclusters.csv")
        res_df.drop(
            res_df.columns[res_df.columns.str.contains("unnamed", case=False)],
            axis=1,
            inplace=True,
        )
        res_df.to_csv(savepath)
        assigned_labels.append(labels[i])
#         if args.verbose:
#             print(f'Sizes of clusters: {", ".join([str((torch.tensor(labels[i]) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')
        
        if gt_ids is not None:
            print(f'Purity: {purity(labels[i], gt_ids):.4f}')
          
        times[i] = (time.clock() - time_start)*20
    cons = consistency(assigned_labels)

    print(f'Consistency: {cons:.4f}\n')
    results['consistency'] = cons
    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in labels])
        amis_mean = np.mean([adjusted_mutual_info_score(gt_ids, x) for x in labels])
        amis_std = np.std([adjusted_mutual_info_score(gt_ids, x) for x in labels])
        ari_mean = np.mean([adjusted_rand_score(gt_ids, x) for x in labels])
        ari_std = np.std([adjusted_rand_score(gt_ids, x) for x in labels])
        fms_mean = np.mean([fowlkes_mallows_score(gt_ids, x) for x in labels])
        fms_std = np.std([fowlkes_mallows_score(gt_ids, x) for x in labels])
        v_mean = np.mean([v_measure_score(gt_ids, x) for x in labels])
        v_std = np.std([v_measure_score(gt_ids, x) for x in labels])
        print(f'Purity: {pur_val_mean:.4f}+-{pur_val_std:.4f}')
        print(f'Purity: {pur_val_mean:.4f}+-{pur_val_std:.4f}')
    time_mean = np.mean(times)
    time_std = np.std(times)
    nll_mean = torch.mean(nlls)
    nll_std = torch.std(nlls)
    print(f'Mean run time: {time_mean:.4f}+-{time_std:.4f}')
    if (args.save_to is not None) and (gt_ids is not None):
        metrics = {
            "Purity": f'{pur_val_mean:.4f}+-{pur_val_std:.4f}' ,
             "AMIS": f'{amis_mean:.4f}+-{amis_std:.4f}' ,
            "ARI": f'{ari_mean:.4f}+-{ari_std:.4f}' ,
            "FMS": f'{fms_mean:.4f}+-{fms_std:.4f}' ,
            "V": f'{v_mean:.4f}+-{v_std:.4f}' ,
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(exp_folder0, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        metrics = {
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)