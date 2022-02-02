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

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.utils import get_logger
from src.utils.metrics import purity

log = get_logger(__name__)


def objective(trial: optuna.trial.Trial, config: DictConfig) -> float:

    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)
    #default_save_dir = config.save_dir
    # Init and prepare lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    thp_dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    thp_dm.prepare_data()
    config.save_dir = str(Path(config.save_dir, "optuna"))
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"Run with Optuna")
    log.info(f"Dataset: {config.data_name}")
    # Init callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "dirpath" in cb_conf:
                cb_conf.dirpath = config.save_dir
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

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
    config.model.num_types = thp_dm.num_events
    # We optimize d_model, d_rnn=d_inner, n_layers=n_head, d_k=d_v
    config.model.d_model = trial.suggest_int("d_model", 64, 512, 64)
    config.model.d_rnn = trial.suggest_int("d_rnn", 256, 2048, 256)
    config.model.d_inner = config.model.d_rnn
    config.model.n_layers = trial.suggest_int("n_layers", 1, 5)
    config.model.n_head = config.model.n_layers
    config.model.d_k = trial.suggest_int("n_layers", 16, 512, 16)
    config.model.d_v = config.model.d_k

    model: LightningModule = hydra.utils.instantiate(config.model)

    # Train the model
    log.info("Starting training")
    trainer.fit(model, thp_dm)

    return trainer.callback_metrics["val_loss"].item()


def thp_optuna(config: DictConfig):
    """
    Optuna module for transformer hawkes hyperparams optimization
    """

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study_with_config = lambda trial: objective(trial, config)
    study.optimize(study_with_config, n_trials=100, timeout=600)

    log.info(f"Number of finished trials: {len(study.trials)}")
    log.info(f"Best trial:")
    trial = study.best_trial
    # write best params to logger and file
    log.info(f" Value: {trial.value}")
    log.info(f" Params: ")
    param_file = open(Path(config.save_dir,"best_params.txt"), "w")
    for key, value in trial.params.items():
        log.info(f"    {key}: {value}")
        param_file.write(f"{key}: {value}\n") 
    param_file.close()
