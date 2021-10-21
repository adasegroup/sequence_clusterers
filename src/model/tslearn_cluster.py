from typing import List

import hydra
import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from pathlib import Path
from src.utils.metrics import consistency, purity

logger = logging.getLogger("tslearn")


def tslearn_infer(config: DictConfig):
    """
    Running tslearn standard method to infer cluster labels
    """

    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    default_save_dir = config.save_dir

    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Run: {i+1}")

        # Init and prepare lightning datamodule
        logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        tslearn_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        tslearn_dm.prepare_data()
        tslearn_dm.setup(stage="test")

        # Init lightning model
        logger.info(f"Instantiating model <{config.model._target_}>")
        config.model.num_clusters = tslearn_dm.num_clusters
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Inference - cluster labels
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=None, logger=None, _convert_="partial"
        )
        logger.info("Starting predicting labels")
        tslearn_dm.setup(stage="test")
        trainer.test(model, tslearn_dm)
        pred_labels = model.final_labels
        gt_labels = tslearn_dm.test_data.target
        # Saving predicted and actual labels - for graphs and tables
        df = pd.DataFrame(columns=["cluster_id", "cluster_tslearn"])
        df["cluster_id"] = gt_labels.tolist()
        df["cluster_tslearn"] = pred_labels.tolist()
        df.to_csv(Path(config.save_dir, "inferredclusters.csv"))
