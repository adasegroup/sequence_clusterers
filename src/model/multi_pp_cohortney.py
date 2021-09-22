__all__ = ["MultiPointProcessSystem"]

from copy import copy

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.networks.losses import cohortney_criterion
from src.networks.lstm_pp import LSTMMultiplePointProcesses
from src.utils import (
    get_learning_rate,
    get_optimizer,
    get_scheduler,
    info_score,
    purity,
)


"""
This file contains system for training clustering with single point process model
"""


class MultiPointProcessSystem(pl.LightningModule):
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
        super(MultiPointProcessSystem, self).__init__()
        self.model = self.create_process_model(hparams)
        self.criterion = cohortney_criterion
        self.batch_size = hparams.batch_size
        self.generator_model = None
        self.params = hparams
        self.automatic_optimization = False
        self.num_m_steps = hparams.get("num_m_steps", 50)
        self.max_computing_size = hparams.get("max_computing_size")
        self.updated_lr = hparams.get("updated_lr")
        self.n_clusters = self.params.n_clusters
        self.true_clusters = hparams.get("true_clusters", 5)
        self.upper_bound_clusters = hparams.get("upper_bound_clusters", 10)
        self.pi = torch.ones(self.n_clusters) / self.n_clusters
        self.epsilon = hparams.get("epsilon", 1e-8)
        self.multiplier_batch = hparams.get("multiplier_batch", 3)
        self.verbose = True

    def setup(self, stage):
        # kwargs = {}
        # self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        # self.val_dataset = dataset(split='val', **kwargs)
        self.N = len(self.train_dataset)
        self.val_dataset = self.train_dataset
        if self.max_computing_size is None:
            self.gamma = torch.zeros(self.params.n_clusters, self.N)
        else:
            self.gamma = torch.zeros(self.params.n_clusters, self.max_computing_size)

    def create_process_model(self, hparams):
        return LSTMMultiplePointProcesses(
            hparams.n_classes + 1,
            hparams.hidden_size,
            hparams.num_layers,
            hparams.n_classes,
            hparams.n_clusters,
            hparams.n_steps,
            dropout=hparams.dropout,
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
            shuffle=False,
            num_workers=self.params.num_workers,
            batch_size=self.batch_size * self.multiplier_batch,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.params.num_workers,
            batch_size=self.batch_size * 20,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        """
        Every step of training consist of single E step and num_m_steps of M-step
        :param batch:
        :param batch_nb:
        :return:
        """
        if self.verbose:
            print("Beginning e-step")

        # E-step
        self.e_step(batch)
        ll = self.m_step(batch)
        # do EVAL ?
        self.micro_eval(batch)
        self.log("lr", get_learning_rate(self.optimizer))
        self.log(
            "train/loss",
            ll.item(),
        )

        return ll

    @torch.no_grad()
    def micro_eval(self, batch):
        # evaluating model
        self.model.eval()
        x, target = batch[:2]
        lambdas = self.model(x)
        gamma = self.compute_gamma(lambdas, x=x, size=(self.n_clusters, len(x)))
        loss = self.criterion(x, lambdas, gamma).item()
        clusters = torch.argmax(gamma, dim=0)
        cluster_partition = 2
        for i in np.unique(clusters.cpu()):
            if self.verbose:
                print(
                    "Cluster",
                    i,
                    ": ",
                    np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                    " with pi = ",
                    self.pi[i],
                )
            cluster_partition = min(
                cluster_partition,
                np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
            )
        if type(target):
            pur = purity(clusters.to("cpu"), target.to("cpu"))
            info = info_score(
                clusters.to("cpu"), target.to("cpu"), len(np.unique(target.to("cpu")))
            )
        else:
            pur = -1
            info = -1
        self.model.train()
        return loss, pur, info

    def validation_step(self, batch, batch_nb):
        """
        :param batch_nb:
        :return:
          val_ll - torch.Tensor, size = (1), log likelihood on validation dataset
                   val_mse - float, mean squared error between obtained lambdas and lambdas of true model on validation,
                             if true model is not provided, then None
        """
        X = batch[0]
        tgt = batch[1]
        lambdas = self.model(X)
        gamma = self.compute_gamma(lambdas, x=X, size=(self.n_clusters, len(X)))
        loss = self.criterion(X, lambdas, gamma).item()
        clusters = torch.argmax(gamma, dim=0)
        # Cluster partition
        cluster_partition = 2
        for i in np.unique(clusters.cpu()):
            if self.verbose:
                print(
                    "Cluster",
                    i,
                    ": ",
                    np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                    " with pi = ",
                    self.pi[i],
                )
            cluster_partition = min(
                cluster_partition,
                np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
            )
        if type(tgt):
            pur = purity(clusters.to("cpu"), tgt)
            info = info_score(clusters.to("cpu"), tgt, len(np.unique(tgt.to("cpu"))))
        else:
            pur = -1
            info = -1

        result = {}
        result["val_ll"] = loss
        result["val_pur"] = pur
        result["val_info"] = info
        result["cluster_partition"] = cluster_partition
        return result

    def validation_epoch_end(self, outputs):
        mean_loss = np.mean([x["val_ll"] for x in outputs])
        mean_pur = np.mean([x["val_pur"] for x in outputs])
        mean_info = np.mean([x["val_info"] for x in outputs])

        self.log("val/loss", mean_loss)
        self.log("val/purity", mean_pur, prog_bar=True)
        self.log("val/info_score", mean_info, prog_bar=True)

    def compute_gamma(self, lambdas, x, size=None, device="cpu"):
        """
        Computes gamma
        :arg
            lambdas - torch.Tensor, size = (batch_size or N, seq_len, number of classes), model output
            x - torch.Tensor, size = (batch_size or N, seq_len, number of classes + 1),
                data, that was processed or None
            size - tuple, gamma size or None
            device - device to compute
        :return
            gamma - torch.Tensor, size = (n_clusters, batch_size or N), probabilities p(k|x_n)
        """
        # preparing gamma template
        if size is None:
            gamma = torch.zeros_like(self.gamma)
        else:
            gamma = torch.zeros(size)

        # preparing delta times and partitions for computing gamma

        dts = x[:, 0, 0].to(device)
        dts = dts[None, :, None, None].to(device)
        partitions = x[:, :, 1:].to(device)
        partitions = partitions[None, :, :, :].to(device)

        # iterations over clusters
        for k in range(self.n_clusters):
            # lambdas of current cluster
            lambdas_k = lambdas[k, :, :, :]
            lambdas_k = lambdas_k[None, :, :, :]

            # weighs in sum
            w = self.pi / self.pi[k]
            w = w[:, None].to(device)

            # computing gamma for k-th cluster
            tmp_sub = (lambdas.to(device) - lambdas_k.to(device)) * dts.to(device)
            tmp = torch.sum(
                -tmp_sub
                + partitions
                * (
                    torch.log(lambdas.to(device) + self.epsilon)
                    - torch.log(lambdas_k.to(device) + self.epsilon)
                ),
                dim=(2, 3),
            )
            tmp = 1 / (torch.sum(w * torch.exp(tmp), dim=0))

            # resolving nans
            tmp[tmp != tmp] = 0

            gamma[k, :] = tmp

        return gamma

    @torch.no_grad()
    def e_step(self, batch):
        """
        Conducts E-step of EM-algorithms, saves the result to self.gamma
        inputs:
                None
        outputs:
                None
        """
        self.model.eval()
        X = batch[0]
        idx = batch[-1]
        with torch.no_grad():
            lambdas = self.model(X)
            self.gamma[:, idx] = self.compute_gamma(
                lambdas, X, size=(self.n_clusters, len(X))
            )
        self.model.train()

    def m_step(self, batch):
        """
        Conducts M-step of EM-algorithm
        :arg
                None
        :return
                log_likelihood_curve - list of floats, losses, obtained during iterations over M-step epochs
                                       and minibatches
                m_step_results - [log_likelihood, purity], mean value of log_likelihood on the last epoch
                                 and purity on the last epoch
                cluster_partitions - float, the minimal value of cluster partition
        """

        # preparing output template
        log_likelihood_curve = []
        ll = []

        # iterations over M-step epochs
        for epoch in range(int(self.num_m_steps)):
            # one epoch training
            ll = self.train_sub_epoch(batch)
            log_likelihood_curve += ll

            # checking for failure WTF
            if np.mean(ll) != np.mean(ll):
                return None, None, None

            # logs
            # if epoch % 10 == 0 and self.verbose:
            print(
                "Loss on sub_epoch {}/{}: {}".format(
                    epoch + 1, self.num_m_steps, np.mean(ll)
                )
            )

        return np.mean(log_likelihood_curve)

    def train_sub_epoch(self, batch_input):
        """
        Conducts one epoch of Neural Net training
        inputs:
                None
        outputs:
                log_likelihood - list of losses obtained during iterations over minibatches
        """
        # preparing random indices
        batch_ids = batch_input[-1]
        # setting model to training and preparing output template
        self.model.train()
        log_likelihood = []
        opts = self.optimizers()
        big_batch, tgt = batch_input[:2]
        indices = np.random.permutation(len(big_batch))

        # iterations over minibatches
        for iteration, start in enumerate(range(0, len(big_batch), self.batch_size)):
            # preparing batch
            cur_ids = indices[start : start + self.batch_size]
            batch = big_batch[cur_ids]

            # one MANUAL step of training
            opts.zero_grad()
            lambdas = self.model(batch)
            loss = self.criterion(batch, lambdas, self.gamma[:, batch_ids[cur_ids]])
            loss.backward()
            opts.step()

            # saving results
            log_likelihood.append(loss.item())

        # if np.mean(log_likelihood) > self.prev_loss:
        #     self.update_checker += 1
        #     if self.update_checker >= self.lr_update_tol:
        #         self.update_checker = 0
        #         lr = 0
        #         for param_group in opts.param_groups:
        #             param_group['lr'] *= self.lr_update_param
        #             lr = param_group['lr']
        #             if self.min_lr is not None:
        #                 if lr < self.min_lr:
        #                     param_group['lr'] = self.updated_lr
        #         if self.min_lr is not None:
        #             if lr < self.min_lr:
        #                 lr = self.updated_lr
        #         self.lr = lr

        # saving previous loss
        self.prev_loss = np.mean(log_likelihood)

        return log_likelihood

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        pass

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs
    ):
        pass
