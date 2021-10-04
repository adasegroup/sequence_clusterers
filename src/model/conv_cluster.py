from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer, LightningDataModule, LightningModule
from pytorch_lightning.loggers import LightningLoggerBase

# from test_tube import Experiment
from pathlib import Path

from src.utils import get_logger

# from src.utils.metrics import consistency, purity

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

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Run: {i+1}")

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

        # inference
        # embeddings = model.predict_step(cohortney_dm.train_data)

        # assigned_labels = []
        # pred_y = model.clusterize(embeddings)
        # if config.aux_module.verbose:
        #    print(
        #        f'Sizes of clusters: {", ".join([str((torch.tensor(pred_y) == i).sum().item()) for i in range(config.num_clusters)])}\n'
        #    )
        # print("preds:", pred_y)
        # torch stacking of predicted labels
        # assigned_labels = torch.LongTensor(assigned_labels)
        # assigned_labels.append(pred_y)

        # gt_ids = cohortney_dm.gt_ids
        # if gt_ids is not None:
        #     print("reals:", gt_ids)
        #     pur = purity(pred_y, gt_ids)
        #     # exp_cae.log({"purity": pur})
        #     print(f"\nPurity: {pur:.4f}")

        # cons = consistency(assigned_labels)

        # print(f"\nConsistency: {cons:.4f}")
        # results = {}
        # results["consistency"] = cons

        # if gt_ids is not None:
        #     pur_val_mean = np.mean([purity(x, gt_ids) for x in assigned_labels])
        #     pur_val_std = np.std([purity(x, gt_ids) for x in assigned_labels])
        #     print(f"Purity: {pur_val_mean}+-{pur_val_std}")
        #     results["purity"] = (pur_val_mean, pur_val_std)
