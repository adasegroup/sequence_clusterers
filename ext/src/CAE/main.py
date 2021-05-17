import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.CAE.litautoencoder import  LitAutoEncoder

from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#from HP import PointProcessStorage, DirichletMixtureModel, EM_clustering
from src.Cohortney.utils import consistency, purity

import src.Cohortney.cohortney as cht
import src.Cohortney.data_utils as du


def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--maxsize', type=int, default=None, help='max number of sequences')
    parser.add_argument('--nmb_cluster', type=int, default=10, help='number of clusters')
    parser.add_argument('--maxlen', type=int, default=-1, help='maximum length of sequence')
    parser.add_argument('--ext', type=str, default='txt', help='extention of files with sequences')
    parser.add_argument('--not_datetime', action='store_true', help='if time values in event sequences are represented in datetime format')
    # hyperparameters for Cohortney
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Tb', type=float, default=7e-6)
    parser.add_argument('--Th', type=float, default=80)
    parser.add_argument('--N', type=int, default=2500)
    parser.add_argument('--n', type=int, default=4, help='n for partition')
    # hyperparameters for training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nruns', type=int, default=1, help='number of trials')
    parser.add_argument('--type', type=str, default=None, help='if it is booking data or not')

    parser.add_argument('--result_path', type=str, help='path to save results')
    args = parser.parse_args()
    return args

np.set_printoptions(threshold=10000)
torch.set_printoptions(threshold=10000)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ss, Ts, class2idx, user_list = du.load_data(Path(args.data_dir), maxsize=args.maxsize, maxlen=args.maxlen, ext=args.ext, datetime=not args.not_datetime, type_=args.type)
    
    gt_ids = None
    if Path(args.data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)
        
    #grid
    grid = []

    #grid generation

    for i in range(1500):
      a = args.gamma**i * args.Tb
      if (a <= args.Th):
        grid.append(a)

      else:
        break

    grid = np.array(grid)

    T_j = grid[-1]
    Delta_T = np.linspace(0, grid[-1], 2**args.n)
    Delta_T = Delta_T[Delta_T< int(T_j)]
    delta_T = tuple(Delta_T)

    _, events_fws_mc = cht.arr_func(user_list, T_j, delta_T,  cht.multiclass_fws_array)
    mc_batch = cht.events_tensor(events_fws_mc)
    
    assigned_labels = []
    for run_id in range(args.nruns):
        print(f'============= RUN {run_id+1} ===============')

        model = LitAutoEncoder(in_channels = mc_batch.shape[1], n_latent_features=16) #

        gpus = [0] if torch.cuda.is_available() else None

#         logger = TensorBoardLogger('lightning_logs', name='my_run')

        trainer = pl.Trainer(
             max_epochs=args.epochs
            , checkpoint_callback=None  #checkpoint_callback
            , logger=None # logger
            # , gradient_clip_val=0.5
            # , precision=16  # On GPU can be beneficial
            , track_grad_norm=2
            , gpus =gpus
            # , fast_dev_run=True
        )

        train_data_batch = torch.utils.data.DataLoader(mc_batch, batch_size=args.batch)
        val_data_batch = torch.utils.data.DataLoader(mc_batch, batch_size=args.batch)

        trainer.fit(model, train_data_batch, val_data_batch)

        ans = model.encoder(mc_batch)
        X = ans.cpu().squeeze().detach().numpy()
        X_trained = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

        results = {}

        kmeans = KMeans(n_clusters=args.nmb_cluster, init='k-means++', max_iter=500, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(X_trained)
    
        assigned_labels.append(pred_y)
        if args.verbose:
            print(f'Sizes of clusters: {", ".join([str((torch.tensor(pred_y) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')
        print ("preds:", pred_y)
        
        pred_y = torch.LongTensor(pred_y)
        if gt_ids is not None:
            print ("reals:", gt_ids)
            pur = purity(pred_y, gt_ids)
            print(f'\nPurity: {pur:.4f}')
    
    assigned_labels = torch.LongTensor(assigned_labels)   
    cons = consistency(assigned_labels)

    print(f'\nConsistency: {cons:.4f}')
    results['consistency'] = cons
    
    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in assigned_labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in assigned_labels])
        print(f'Purity: {pur_val_mean}+-{pur_val_std}')
        results['purity'] = (pur_val_mean, pur_val_std)


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)
