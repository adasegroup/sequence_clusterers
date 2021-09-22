from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase
from test_tube import Experiment
from tslearn.clustering import TimeSeriesKMeans, KShape

from src.utils import get_logger
from src.utils.metrics import consistency, purity

log = get_logger(__name__)


def tslearn_infer(config: DictConfig):
    """
    Employing tslearn standard method to infer cluster labels
    """

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

    # Init and prepare lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    tslearn_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    tslearn_dm.prepare_data()
    tslearn_dm.setup(stage="test")

    # inference
    if config.aux_module.modelname == "k_shape":
        cluster_model = KShape(
            n_clusters=config.num_clusters, max_iter=config.aux_module.max_iter
        )
    elif config.aux_module.modelname == "k_means_softdtw":
        cluster_model = TimeSeriesKMeans(
            n_clusters=config.num_clusters,
            metric=config.aux_module.kmeans.metric,
            max_iter=config.aux_module.max_iter,
        )

    assigned_labels = []
    pred_y = cluster_model.fit_predict(tslearn_dm.test_data)
    if config.aux_module.verbose:
        print(
            f'Sizes of clusters: {", ".join([str((torch.tensor(pred_y) == i).sum().item()) for i in range(config.num_clusters)])}\n'
        )
    print("preds:", pred_y)
    pred_y = torch.LongTensor(pred_y)
    # torch stacking of predicted labels
    # assigned_labels = torch.LongTensor(assigned_labels)
    assigned_labels.append(pred_y)

    gt_ids = tslearn_dm.gt_ids
    if gt_ids is not None:
        print("reals:", gt_ids)
        pur = purity(pred_y, gt_ids)
        exp_cae.log({"purity": pur})
        print(f"\nPurity: {pur:.4f}")

    cons = consistency(assigned_labels)

    print(f"\nConsistency: {cons:.4f}")
    results = {}
    results["consistency"] = cons

    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in assigned_labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in assigned_labels])
        print(f"Purity: {pur_val_mean}+-{pur_val_std}")
        results["purity"] = (pur_val_mean, pur_val_std)

    exp_cae.save()
    exp_cae.close()
