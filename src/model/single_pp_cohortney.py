__all__ = ["SinglePointProcessSystem"]


"""
This file contains system for training clustering with single point process model
"""
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.networks.losses import cohortney_criterion
from src.networks.lstm_pp import LSTMSinglePointProcess
from src.utils import get_learning_rate, get_optimizer, get_scheduler


class SinglePointProcessSystem(pl.LightningModule):
    """
    Trainer for single point process model
    """

    def __init__(self, hparams):
        """
        :arg:
              process - torch.nn.Module, model to train
              optimizer - optimizer to train model
              criterion - loss to optimize, takes batch, lambdas, dts
              x - torch.Tensor, training data
              val - torch.Tensor, validation data
              max_epochs - int, number of epochs for sgd training
              batch_size - int, batch size
              generator_model - torch.nn.Module, true model, that was used for generation or None
        :return:
              the same as inputs
        """
        super(SinglePointProcessSystem, self).__init__()
        self.model = self.create_process_model(hparams)
        self.criterion = cohortney_criterion
        self.batch_size = hparams.batch_size
        self.params = hparams
        self.generator_model = None  # creat_generator_model(hparams)

    def setup(self, stage):
        self.N = len(self.train_dataset)

    def create_process_model(self, hparams):
        return LSTMSinglePointProcess(
            hparams.n_classes + 1,
            hparams.hidden_size,
            hparams.num_layers,
            hparams.n_classes,
        )

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.params.optimizer, self.model)
        scheduler = get_scheduler(self.params.scheduler, self.optimizer)
        return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.params.num_workers,
            batch_size=self.params.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.params.num_workers,
            batch_size=self.params.batch_size,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        res = {}
        x, tgt = batch
        lambdas = self.model(x)
        loss = self.criterion(x, lambdas, tgt)

        # saving results
        if self.generator_model:
            true_lambdas = self.generator_model(batch)
            res["mse"] = np.var(
                (lambdas.detach().numpy() - true_lambdas.detach().numpy())
            )

        self.log("lr", get_learning_rate(self.optimizer))
        self.log(
            "train/loss",
            loss.item(),
        )

        self.log("train/mse", res.pop("mse", 0.0), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        """
        :param batch_nb:
        :return:
          val_ll - torch.Tensor, size = (1), log likelihood on validation dataset
                   val_mse - float, mean squared error between obtained lambdas and lambdas of true model on validation,
                             if true model is not provided, then None
        """
        x, tgt = batch
        lambdas = self.model(x)
        result = {}

        result["val_ll"] = self.criterion(x, lambdas, tgt)
        if self.generator_model:
            true_lambdas = self.generator_model(batch)
            result["val_mse"] = np.var(
                (lambdas.detach().numpy() - true_lambdas.detach().numpy())
            )
        else:
            result["val_mse"] = torch.tensor(1.0)

        return result

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_ll"] for x in outputs]).mean()
        mean_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        #
        self.log("val/loss", mean_loss)
        self.log("val/mse", mean_mse, prog_bar=True)
