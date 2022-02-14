from typing import Any, List, Optional

import torch
from pytorch_lightning import LightningModule
from src.utils.losses import loss_getter
from src.utils.metrics import purity
from src.utils.model_helpers import model_getter
from src.utils.file_system_utils import save_model
import numpy as np
import time

class LALModel(LightningModule):
    """
    LightningModule for LAL clustering.
    """

    def __init__(
            self,
            model_type: str = 'LSTM',
            model_params: dict = {'input_size': 6,
                                  'hidden_size': 128,
                                  'num_layers': 3,
                                  'num_classes': 5,
                                  'num_clusters': 1,
                                  'n_steps': 128,
                                  'dropout': 0.3},
            n_clusters: int = 1,
            epsilon: float = 1e-8,
            weight_decay: float = 1e-5,
            lr: float = 1e-3,
            save_dir: Optional[str] = None
    ):
        # TODO saving model
        # TODO make model an instance
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.n_clusters = n_clusters

        self.model = model_getter(model_type, model_params)

        # loss function
        self.criterion = loss_getter(model_type)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = purity
        self.val_metric = purity
        self.test_metric = purity
        self.prev_loss = 1e9
        self.cur_loss = 1e9
        self.final_labels = None
        
        self.start_time = time.time()

    def forward(self, x: torch.Tensor):
        lambdas, gamma = self.model(x, self.hparams.epsilon, self.device)
        return lambdas, gamma

    def step(self, batch: Any):
        x, y, gamma = batch
        gamma = gamma.T
        lambdas, proba = self.model(x, self.hparams.epsilon, self.device)
        loss = self.criterion(x, lambdas, gamma, self.hparams.epsilon)
        preds = torch.argmax(proba, dim=0)
        return loss, preds, y

    def self_gamma_step(self, batch: Any):
        x, y, gamma = batch
        lambdas, proba = self.model(x,  self.hparams.epsilon, self.device)
        loss = self.criterion(x, lambdas, proba, self.hparams.epsilon)
        preds = torch.argmax(proba, dim=0)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        pur = self.train_metric(preds, targets)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "pur": pur}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        cur_losses = []
        cur_purs = []
        for batch_results in outputs:
            cur_losses.append(float(batch_results["loss"].cpu()))
            cur_purs.append(float(batch_results["pur"]))
        self.prev_loss = self.cur_loss
        self.cur_loss = np.mean(cur_losses)
        
        cur_time = time.time()

        self.log("train/loss", np.mean(cur_losses), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/pur", np.mean(cur_purs), on_step=False, on_epoch=True, prog_bar=True)
        self.log("time", cur_time - self.start_time, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        pur = self.val_metric(preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets, "pur": pur}

    def validation_epoch_end(self, outputs: List[Any]):
        cur_losses = []
        cur_purs = []
        for batch_results in outputs:
            cur_losses.append(float(batch_results["loss"].cpu()))
            cur_purs.append(float(batch_results["pur"]))

        self.log("val/loss", np.mean(cur_losses), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/pur", np.mean(cur_purs), on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.save_dir:
            save_model(self.model, self.hparams.save_dir)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.final_labels = preds

        # log test metrics
        pur = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/pur", pur, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        if self.hparams.save_dir:
            save_model(self.model, self.hparams.save_dir)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
