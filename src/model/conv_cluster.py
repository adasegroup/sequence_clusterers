from typing import List

import hydra
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from test_tube import Experiment
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.cluster import KMeans

from src.utils import get_logger
from src.networks import Conv1dAutoEncoder
from src.utils import make_grid
from src.utils.cohortney_utils import arr_func, events_tensor, multiclass_fws_array
from src.utils.datamodule import load_data
from src.utils.metrics import consistency, purity

log = get_logger(__name__)


def cae_train(config: DictConfig):
    """
        Train module for convolutional autoencoder clustering for event sequences
        every event first pass through pure cohortney module
        then passed into encoder and clustered using KMeans over conv1d codes
    """

    args = config.aux_module
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)
    
    exp_cae = Experiment(config.logger.test_tube.save_dir, config.logger.test_tube.name +'/'+ config.data_dir.split('/')[-1])

    ss, Ts, class2idx, user_list = load_data(Path(config.data_dir), maxsize=args.maxsize, maxlen=args.maxlen,
                                             ext=args.ext, datetime=not args.not_datetime, type_=args.type)

    gt_ids = None
    if Path(args.data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)

    # grid generation
    grid = make_grid(args.gamma, args.Tb, args.Th, args.N, args.n)

    T_j = grid[-1]
    Delta_T = np.linspace(0, grid[-1], 2 ** args.n)
    Delta_T = Delta_T[Delta_T < int(T_j)]
    delta_T = tuple(Delta_T)

    _, events_fws_mc = arr_func(user_list, T_j, delta_T, multiclass_fws_array)
    mc_batch = events_tensor(events_fws_mc)

    assigned_labels = []
    model = Conv1dAutoEncoder(in_channels=mc_batch.shape[1],
                              n_latent_features=16)  #


    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))


    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=None, logger=None, _convert_="partial"
    )

    
    train_data_batch = DataLoader(mc_batch, batch_size=args.batch)
    val_data_batch = DataLoader(mc_batch, batch_size=args.batch)

    trainer.fit(model, train_data_batch, val_data_batch)
    
    ans = model.encoder(mc_batch)
    X = ans.cpu().squeeze().detach().numpy()
    X_trained = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    results = {}

    kmeans = KMeans(n_clusters=args.nmb_cluster, init='k-means++', max_iter=500, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X_trained)

    assigned_labels.append(pred_y)
    if args.verbose:
        print(
            f'Sizes of clusters: {", ".join([str((torch.tensor(pred_y) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')
    print("preds:", pred_y)

    pred_y = torch.LongTensor(pred_y)
    if gt_ids is not None:
        print("reals:", gt_ids)
        pur = purity(pred_y, gt_ids)
        exp_cae.log({'purity': pur})
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
        
    exp_cae.save()
    exp_cae.close()
