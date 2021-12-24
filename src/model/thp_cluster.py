# from test_tube import Experiment
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import get_logger
from src.utils.metrics import purity

log = get_logger(__name__)


def thp_train(config: DictConfig):
    """
    Training module for transformer hawkes clustering for event sequences
    """
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)
    default_save_dir = config.save_dir
    # Init and prepare lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    thp_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    thp_dm.prepare_data()

    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Run: {i+1}")
        log.info(f"Dataset: {config.data_name}")
        # Init callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "dirpath" in cb_conf:
                    cb_conf.dirpath = config.save_dir
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
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

        thp_dm.setup(stage="fit")
        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        config.model.num_clusters = thp_dm.num_clusters
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Train the model
        log.info("Starting training")
        trainer.fit(model, thp_dm)
        # Inference - cluster labels
        log.info("Starting predicting labels")
        thp_dm.setup(stage="test")
        trainer.test(model, thp_dm)
        pred_labels = model.final_labels
        gt_labels = thp_dm.test_data.target
        # Saving predicted and actual labels - for graphs and tables
        df = pd.DataFrame(columns=["cluster_id", "cluster_thp"])
        df["cluster_id"] = gt_labels.tolist()
        df["cluster_thp"] = pred_labels.tolist()
        df.to_csv(Path(config.save_dir, "inferredclusters.csv"))
