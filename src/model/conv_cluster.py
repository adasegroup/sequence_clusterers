# from test_tube import Experiment
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import get_logger
from src.utils.metrics import purity

log = get_logger(__name__)


def cae_train(config: DictConfig):
    """
    Training module for convolutional autoencoder clustering for event sequences
    every event first passes through pure cohortney module
    then - to encoder, and finally clustered using KMeans over conv1d representations
    """
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)
    default_save_dir = config.save_dir


    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Run: {i+1}")
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
                    # if "save_dir" in lg_conf:
                    #     lg_conf.save_dir = config.save_dir
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    # lg_conf.version = "exp_" + str(i)
                    logger.append(hydra.utils.instantiate(lg_conf))

        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

        # Init and prepare lightning datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        cohortney_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        cohortney_dm.prepare_data()
        cohortney_dm.setup(stage="fit")

        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        config.model.in_channels = cohortney_dm.train_data.data.shape[1]
        config.model.num_clusters = config.num_clusters
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Train the model
        log.info("Starting training!")
        trainer.fit(model, cohortney_dm)

        # Inference - to be restructed into the model
        log.info("Starting predicting labels")
        cohortney_dm.setup(stage="test")
        trainer.test(model, cohortney_dm)
        pred_labels = model.final_labels
        gt_labels = cohortney_dm.test_data.target
        # saving predicted and actual labels - for graphs and tables
        df = pd.DataFrame(columns=["cluster_id", "cluster_cohortney"])
        df["cluster_id"] = gt_labels.tolist()
        df["cluster_cohortney"] = pred_labels.tolist()
        df.to_csv(Path(config.save_dir, "inferredclusters.csv"))
