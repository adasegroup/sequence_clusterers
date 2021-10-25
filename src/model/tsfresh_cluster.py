from typing import List

import hydra
import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from pathlib import Path

logger = logging.getLogger("tsfresh")


def tsfresh_infer(config: DictConfig):
    """
    Running tslearn standard method to infer cluster labels
    """

    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    default_save_dir = config.save_dir
    # Init and prepare lightning datamodule
    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    tsfresh_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    tsfresh_dm.prepare_data()
    tsfresh_dm.setup(stage="test")

    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset: {config.data_name}")
        logger.info(f"Run: {i+1}")

        # Init lightning model
        logger.info(f"Instantiating model <{config.model._target_}>")
        config.model.num_clusters = tsfresh_dm.num_clusters
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Inference - cluster labels
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=None, logger=None, _convert_="partial"
        )
        logger.info("Starting predicting labels")
        trainer.test(model, tsfresh_dm)
        pred_labels = model.final_labels
        gt_labels = tsfresh_dm.test_data.target
        # Saving predicted and actual labels - for graphs and tables
        df = pd.DataFrame(columns=["cluster_id", "cluster_tsfresh"])
        df["cluster_id"] = gt_labels.tolist()
        df["cluster_tsfresh"] = pred_labels.tolist()
        df.to_csv(Path(config.save_dir, "inferredclusters.csv"))
