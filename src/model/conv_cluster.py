from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.cluster import KMeans
from test_tube import Experiment

from src.datamodules.datamodule import CohortneyDataModule
from src.networks import Conv1dAutoEncoder
from src.utils import get_logger
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

    exp_cae = Experiment(
        config.logger.test_tube.save_dir,
        config.logger.test_tube.name + "/" + config.data_dir.split("/")[-1],
    )

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

    cohortney_dm = CohortneyDataModule(args, config.data_dir)
    cohortney_dm.prepare_data()
    cohortney_dm.setup(stage="fit")

    model = Conv1dAutoEncoder(
        in_channels=cohortney_dm.train_data.shape[1], n_latent_features=16
    )
    trainer.fit(model, cohortney_dm)

    # inference - to be refactored
    ans = model.encoder(cohortney_dm.train_data)
    X = ans.cpu().squeeze().detach().numpy()
    X_trained = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    results = {}
    assigned_labels = []

    kmeans = KMeans(
        n_clusters=args.nmb_cluster,
        init="k-means++",
        max_iter=500,
        n_init=10,
        random_state=0,
    )
    pred_y = kmeans.fit_predict(X_trained)

    assigned_labels.append(pred_y)
    if args.verbose:
        print(
            f'Sizes of clusters: {", ".join([str((torch.tensor(pred_y) == i).sum().item()) for i in range(args.nmb_cluster)])}\n'
        )
    print("preds:", pred_y)

    pred_y = torch.LongTensor(pred_y)
    gt_ids = cohortney_dm.gt_ids
    if gt_ids is not None:
        print("reals:", gt_ids)
        pur = purity(pred_y, gt_ids)
        exp_cae.log({"purity": pur})
        print(f"\nPurity: {pur:.4f}")

    assigned_labels = torch.LongTensor(assigned_labels)
    cons = consistency(assigned_labels)

    print(f"\nConsistency: {cons:.4f}")
    results["consistency"] = cons

    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in assigned_labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in assigned_labels])
        print(f"Purity: {pur_val_mean}+-{pur_val_std}")
        results["purity"] = (pur_val_mean, pur_val_std)

    exp_cae.save()
    exp_cae.close()
