"""
    This file contains trainers, that conduct training of the model according to considered methods
"""
import os
import optuna
import time
import numpy as np
from utils.metrics import purity, info_score
import torch
import math
from sklearn.cluster import KMeans
from models.LSTM import LSTMMultiplePointProcesses


class TrainerSingle:
    """
    Trainer for single point process model
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        x,
        val,
        max_epochs=100,
        batch_size=30,
        generator_model=None,
    ):
        """
        input:
              model - torch.nn.Module, model to train
              optimizer - optimizer to train model
              criterion - loss to optimize, takes batch, lambdas, dts
              x - torch.Tensor, training data
              val - torch.Tensor, validation data
              max_epochs - int, number of epochs for sgd training
              batch_size - int, batch size
              generator_model - torch.nn.Module, true model, that was used for generation or None
        model parameters:
              the same as inputs
        """
        self.N = x.shape[0]
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.x = x
        self.val = val
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.generator_model = generator_model

    def train_epoch(self):
        """
        Conducts one epoch training

        input:
               None

        output:
               log_likelihood - list, list of all losses obtained during batch iterations
               mse - list, list of mean squared errors between obtained lambdas and lambdas of true model,
                     if true model is not provided, then the list is empty
               val_ll - torch.Tensor, size = (1), log likelihood on validation dataset
               val_mse - float, mean squared error between obtained lambdas and lambdas of true model on validation,
                         if true model is not provided, then None
        """
        indices = np.random.permutation(self.N)
        self.model.train()

        # initializing outputs
        log_likelihood = []
        mse = []
        val_mse = None

        # iterations over minibatches
        for iteration, start in enumerate(
            range(0, self.N - self.batch_size, self.batch_size)
        ):
            batch_ids = indices[start : start + self.batch_size]
            batch = self.x[batch_ids]

            # optimization
            self.optimizer.zero_grad()
            lambdas = self.model(batch)
            loss = self.criterion(batch, lambdas, batch[:, 0, 0])
            loss.backward()
            self.optimizer.step()

            # saving results
            log_likelihood.append(loss.item())
            if self.generator_model:
                true_lambdas = self.generator_model(batch)
                mse.append(
                    np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy()))
                )

        # validation
        self.model.eval()
        lambdas = self.model(self.val)
        val_ll = self.criterion(self.val, lambdas, self.val[:, 0, 0])
        if self.generator_model:
            true_lambdas = self.generator_model(self.val)
            val_mse = np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy()))

        return log_likelihood, mse, val_ll, val_mse

    def train(self):
        """
        Conducts training

        input:
               None

        output:
               losses - list, list of all mean log likelihoods obtained during training on each epoch
               val_losses - list, list of all log likelihoods obtained during training on each epoch on validation
               mses - list, list of all mean squared errors between obtained lambdas and true lambdas on each epoch
               val_mses - list, the same but on validation
        """
        self.generator_model.eval()

        # initializing outputs
        losses = []
        val_losses = []
        mses = []
        val_mses = []

        # iterations over epochs
        for epoch in range(self.max_epochs):
            ll, mse, val_ll, val_mse = self.train_epoch()
            losses.append(np.mean(ll))
            val_losses.append(val_ll)
            mses.append(np.mean(mse))
            val_mses.append(val_mse)

            # logs
            if len(mse):
                print(
                    "On epoch {}/{}, ll = {}, mse = {}, val_ll = {}, val_mse = {}".format(
                        epoch,
                        self.max_epochs,
                        np.mean(ll),
                        np.mean(mse),
                        val_ll,
                        val_mse,
                    )
                )
            else:
                print(
                    "On epoch {}/{}, ll = {}, val_ll = {}".format(
                        epoch, self.max_epochs, np.mean(ll), val_ll
                    )
                )

        return losses, val_losses, mses, val_mses


class TrainerClusterwise:
    """
    Trainer for multiple point processes clustering
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        data,
        n_clusters,
        exper_path,
        target=None,
        epsilon=1e-8,
        max_epoch=50,
        max_m_step_epoch=50,
        weight_decay=1e-5,
        lr=1e-3,
        lr_update_tol=25,
        lr_update_param=0.5,
        random_walking_max_epoch=40,
        true_clusters=5,
        upper_bound_clusters=10,
        min_lr=None,
        updated_lr=None,
        batch_size=150,
        verbose=False,
        best_model_path=None,
        max_computing_size=None,
        full_purity=True,
        trial=None
    ):
        """
        inputs:
                model - torch.nn.Module, model to train
                optimizer - optimizer used for training
                device - device, that is used for training
                data - torch.Tensor, size = (N, sequence length, number of classes + 1),
                       partitions of the point processes
                n_clusters - int, initial number of different point processes
                exper_path - path to experiment folder
                target - torch.Tensor, size = (N), true labels or None
                epsilon - float, used for log-s regularization log(x) -> log(x + epsilon)
                max_epoch - int, number of epochs of EM algorithm
                max_m_step_epoch - float, number of epochs of neural net training on M-step
                lr_update_tol - int, tolerance before updating learning rate
                lr_update_param - float, learning rate multiplier
                random_walking_max_epoch - int, number of epochs when random walking of the number of clusters
                                           is available
                true_clusters - int, true number of clusters
                upper_bound_clusters - int, upper bound of the number of clusters
                min_lr - float - minimal lr value, when achieved lr is updated to updated_lr and update params set
                         to default
                updated_lr - float, lr after achieving min_lr
                batch_size - int, batch size during neural net training
                verbose - bool, if True, provides info during training
                best_model_path - str, where the best model according to loss should be saved or None
                max_computing_size - int, if not None, then constraints gamma size (one EM-algorithm step)
                full_purity - bool, if True, purity is computed on all dataset
                trial - optuna study object

        parameters:
                N - int, number of data points
                model - torch.nn.Module, model to train
                optimizer - optimizer used for training
                device - device used for training
                X - torch.Tensor, size = (N, sequence length, number of classes + 1),
                    partitions of the point processes
                target - torch.Tensor, size = (N), true labels or None
                n_clusters - int, number of different point processes
                max_epoch - int, number of epochs of EM algorithm
                lr_update_tol - int, tolerance before updating learning rate
                update_checker - int, checker, that is compared to tolerance, increased by one every time loss is
                                 greater then on the previous iteration
                lr_update_param - float, learning rate multiplier
                random_walking_max_epoch - int, number of epochs when random walking of the number of clusters
                                           is available
                true_clusters - int, true number of clusters
                upper_bound_clusters - int, upper bound of the number of clusters
                min_lr - float - minimal lr value, when achieved lr is updated to updated_lr and update params set
                         to default
                updated_lr - float, lr after achieving min_lr
                epsilon - float, used for log-s regularization log(x) -> log(x + epsilon)
                prev_loss - float, loss on previous iteration, used for updating update_checker
                batch_size - int, batch size during neural net training
                pi - torch.Tensor, size = (n_clusters), mixing coefficients, here are fixed and equal 1/n_clusters
                gamma - torch.Tensor, size = (n_clusters, number of data points), probabilities p(k|x_n)
                best_model_path - str, where the best model according to loss should be saved or None
                prev_loss_model - float, loss obtained for the best model
                max_computing_size - int, if not None, then constraints gamma size (one EM-algorithm step)
                fool_purity - bool, if True, purity is computed on all dataset
        """
        self.N = data.shape[0]
        self.model = model
        self.optimizer = optimizer
        self.device = device
        if max_computing_size is None:
            self.X = data.to(device)
            if target is not None:
                self.target = target.to(device)
            else:
                self.target = None
        else:
            self.X = data
            if target is not None:
                self.target = target
            else:
                self.target = None
        self.n_clusters = n_clusters
        self.exper_path = exper_path
        self.max_epoch = max_epoch
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_update_tol = lr_update_tol
        self.min_lr = min_lr
        self.updated_lr = updated_lr
        self.update_checker = -1
        self.epsilon = epsilon
        self.lr_update_param = lr_update_param
        self.prev_loss = 0
        self.max_m_step_epoch = max_m_step_epoch
        self.batch_size = batch_size
        self.pi = (torch.ones(n_clusters) / n_clusters).to(device)
        if max_computing_size is None:
            self.gamma = torch.zeros(n_clusters, self.N).to(device)
        else:
            self.gamma = torch.zeros(n_clusters, max_computing_size).to(device)
        self.max_computing_size = max_computing_size
        self.verbose = verbose
        self.best_model_path = best_model_path
        self.prev_loss_model = 0
        self.full_purity = full_purity
        self.start_time = time.time()
        self.random_walking_max_epoch = random_walking_max_epoch
        self.true_clusters = true_clusters
        self.upper_bound_clusters = upper_bound_clusters
        self.trial = trial

    def loss(self, partitions, lambdas, gamma):
        """
        Computes loss

        inputs:
                partitions - torch.Tensor, size = (batch_size, seq_len, number of classes + 1)
                lambdas - torch.Tensor, size = (batch_size, seq_len, number of classes), model output
                gamma - torch.Tensor, size = (n_clusters, batch_size), probabilities p(k|x_n)

        outputs:
                loss - torch.Tensor, size = (1), sum of output log likelihood weighted with convoluted gamma
                       and prior distribution log likelihood
        """
        # computing poisson parameters
        dts = partitions[:, 0, 0].to(self.device)
        dts = dts[None, :, None, None].to(self.device)
        tmp = lambdas * dts

        # preparing partitions
        p = partitions[None, :, :, 1:].to(self.device)

        # computing log likelihoods of every timestamp
        tmp1 = tmp - p * torch.log(tmp + self.epsilon) + torch.lgamma(p + 1)

        # computing log likelihoods of data points
        tmp2 = torch.sum(tmp1, dim=(2, 3))

        # computing loss
        tmp3 = gamma.to(self.device) * tmp2
        loss = torch.sum(tmp3)

        return loss

    def compute_gamma(self, lambdas, x=None, size=None, device="cpu"):
        """
        Computes gamma

        inputs:
                lambdas - torch.Tensor, size = (batch_size or N, seq_len, number of classes), model output
                x - torch.Tensor, size = (batch_size or N, seq_len, number of classes + 1),
                    data, that was processed or None
                size - tuple, gamma size or None
                device - device to compute

        outputs:
                gamma - torch.Tensor, size = (n_clusters, batch_size or N), probabilities p(k|x_n)
        """
        # preparing gamma template
        if size is None:
            gamma = torch.zeros_like(self.gamma)
        else:
            gamma = torch.zeros(size)

        # preparing delta times and partitions for computing gamma
        if x is None:
            dts = self.X[:, 0, 0].to(device)
            dts = dts[None, :, None, None].to(device)
            partitions = self.X[:, :, 1:].to(device)
            partitions = partitions[None, :, :, :].to(device)
        else:
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

    def get_gamma_stats(self):
        """
        Obtains gamma (probabilities) stats

        inputs:
                None

        outputs:
                stats - dict:
                              stats['min'] - minimal probability per cluster
                              stats['max'] - maximal probability per cluster
                              stats['min_main'] - minimal probability of predicted cluster
                              stats['max_main'] - maximal probability of predicted cluster
                              stats['mean_main'] - mean probability of predicted cluster
                              stats['std_main'] - std of probabilities of predicted cluster
                              stats['median_main'] - median of probabilities of predicted cluster
        """
        stats = dict()

        # computing stats
        stats["min"] = torch.min(self.gamma, dim=1).values
        stats["max"] = torch.max(self.gamma, dim=1).values
        stats["min_main"] = torch.min(torch.max(self.gamma, dim=0).values)
        stats["max_main"] = torch.max(torch.max(self.gamma, dim=0).values)
        stats["mean_main"] = torch.mean(torch.max(self.gamma, dim=0).values)
        stats["std_main"] = torch.std(torch.max(self.gamma, dim=0).values)
        stats["median_main"] = torch.median(torch.max(self.gamma, dim=0).values)

        return stats

    def get_model_stats(self):
        """
        Obtains model parameters stats

        inputs:
                None

        outputs:
                stats - list of dicts:
                              stats[i]['min'] - minimal value in weighs of i-th parameter
                              stats[i]['max'] - maximal value in weighs of i-th parameter
                              stats[i]['mean'] - mean value of weighs of i-th parameter
                              stats[i]['std'] - std of values of weighs of i-th parameter
                              stats[i]['median'] - median of values of weighs of i-th parameter
        """
        stats = []

        # iterations over model parameters
        for param in self.model.parameters():
            sub_stats = dict()
            sub_stats["min"] = torch.min(param.data)
            sub_stats["max"] = torch.max(param.data)
            sub_stats["mean"] = torch.mean(param.data)
            sub_stats["std"] = torch.std(param.data)
            sub_stats["median"] = torch.median(param.data)
            stats.append(sub_stats)

        return stats

    @staticmethod
    def get_lambda_stats(lambdas):
        """
        Obtains lambda stats

        inputs:
                lambdas - torch.Tensor, size = (batch_size or N, seq_len, number of classes), model output

        outputs;
                stats - dict:
                              stats['min'] - minimal value of lambdas in cluster for each type of event
                              stats['max'] - maximal value of lambdas in cluster for each type of event
                              stats['mean'] - mean value of lambdas in each cluster for each type of event
                              stats['std'] - std of values of lambdas in each cluster for each type of event
        """
        stats = dict()

        # computing stats
        stats["min"] = torch.min(lambdas, dim=1).values
        stats["min"] = torch.min(stats["min"], dim=1).values
        stats["max"] = torch.max(lambdas, dim=1).values
        stats["max"] = torch.max(stats["max"], dim=1).values
        stats["mean"] = torch.mean(lambdas, dim=(1, 2))
        stats["std"] = torch.std(lambdas, dim=(1, 2))

        return stats

    def e_step(self, ids=None):
        """
        Conducts E-step of EM-algorithms, saves the result to self.gamma

        inputs:
                None

        outputs:
                None
        """
        self.model.eval()
        with torch.no_grad():
            if ids is None:
                lambdas = self.model(self.X)
                self.gamma = self.compute_gamma(lambdas)
            else:
                lambdas = self.model(self.X[ids].to(self.device))
                self.gamma = self.compute_gamma(
                    lambdas, x=self.X[ids], size=(self.n_clusters, len(ids))
                )

    def train_epoch(self, big_batch=None):
        """
        Conducts one epoch of Neural Net training

        inputs:
                None

        outputs:
                log_likelihood - list of losses obtained during iterations over minibatches
        """
        # preparing random indices
        if self.max_computing_size is None:
            indices = np.random.permutation(self.N)
        else:
            indices = np.random.permutation(self.max_computing_size)

        # setting model to training and preparing output template
        self.model.train()
        log_likelihood = []

        # iterations over minibatches
        for iteration, start in enumerate(
            range(
                0,
                (self.N if self.max_computing_size is None else self.max_computing_size)
                - self.batch_size,
                self.batch_size,
            )
        ):
            # preparing batch
            batch_ids = indices[start : start + self.batch_size]
            if self.max_computing_size is None:
                batch = self.X[batch_ids].to(self.device)
            else:
                batch = big_batch[batch_ids].to(self.device)

            # one step of training
            self.optimizer.zero_grad()
            lambdas = self.model(batch).to(self.device)
            loss = self.loss(batch, lambdas, self.gamma[:, batch_ids])
            loss.backward()
            self.optimizer.step()

            # saving results
            log_likelihood.append(loss.item())

        if np.mean(log_likelihood) > self.prev_loss:
            self.update_checker += 1
            if self.update_checker >= self.lr_update_tol:
                self.update_checker = 0
                lr = 0
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.lr_update_param
                    lr = param_group["lr"]
                    if self.min_lr is not None:
                        if lr < self.min_lr:
                            param_group["lr"] = self.updated_lr
                if self.min_lr is not None:
                    if lr < self.min_lr:
                        lr = self.updated_lr
                self.lr = lr

        # saving previous loss
        self.prev_loss = np.mean(log_likelihood)

        return log_likelihood

    def m_step(self, big_batch=None, ids=None):
        """
        Conducts M-step of EM-algorithm

        inputs:
                None

        outputs:
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
        for epoch in range(int(self.max_m_step_epoch)):
            # one epoch training
            ll = self.train_epoch(big_batch=big_batch)
            log_likelihood_curve += ll

            # checking for failure
            if np.mean(ll) != np.mean(ll):
                return None, None, None

            # logs
            if epoch % 10 == 0 and self.verbose:
                print(
                    "Loss on sub_epoch {}/{}: {}".format(
                        epoch + 1, self.max_m_step_epoch, np.mean(ll)
                    )
                )

        # evaluating model
        self.model.eval()
        with torch.no_grad():
            if (self.max_computing_size is None) or self.full_purity:
                lambdas = self.model(self.X.to(self.device))
                gamma = self.compute_gamma(
                    lambdas, x=self.X, size=(self.n_clusters, self.N)
                )
                loss = self.loss(
                    self.X.to(self.device),
                    lambdas.to(self.device),
                    gamma.to(self.device),
                ).item()
            else:
                lambdas = self.model(big_batch)
                gamma = self.compute_gamma(
                    lambdas,
                    x=big_batch,
                    size=(self.n_clusters, self.max_computing_size),
                )
                loss = self.loss(
                    big_batch.to(self.device),
                    lambdas.to(self.device),
                    gamma.to(self.device),
                ).item()
            clusters = torch.argmax(gamma, dim=0)
            if self.verbose:
                print("Cluster partition")
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
            if self.target is not None:
                pur = purity(
                    clusters.to("cpu"),
                    self.target[ids]
                    if (ids is not None) and (not self.full_purity)
                    else self.target.to("cpu"),
                )
                info = info_score(
                    clusters.to("cpu"),
                    self.target[ids]
                    if (ids is not None) and (not self.full_purity)
                    else self.target.to("cpu"),
                    len(np.unique(self.target.to("cpu"))),
                )
            else:
                pur = -1
                info = -1

        return log_likelihood_curve, [loss, pur, info], cluster_partition

    def compute_ll(self, big_batch, ids, to_print):
        if (self.max_computing_size is None) or self.full_purity:
            lambdas = self.model(self.X.to(self.device))
            gamma = self.compute_gamma(
                lambdas, x=self.X, size=(self.n_clusters, self.N)
            )
            ll = self.loss(
                self.X.to(self.device), lambdas.to(self.device), gamma.to(self.device)
            ).item()
        else:
            lambdas = self.model(big_batch)
            gamma = self.compute_gamma(
                lambdas, x=big_batch, size=(self.n_clusters, self.max_computing_size)
            )
            ll = self.loss(
                big_batch.to(self.device),
                lambdas.to(self.device),
                gamma.to(self.device),
            ).item()

        clusters = torch.argmax(gamma, dim=0)
        if self.verbose:
            print("Cluster partition")
        cluster_partition = 2
        for i in np.unique(clusters.cpu().detach()):
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
        if self.target is not None:
            pur = purity(
                clusters.to("cpu"),
                self.target[ids]
                if (ids is not None) and (not self.full_purity)
                else self.target.to("cpu"),
            )
        else:
            pur = None
        if self.verbose:
            print("{} loss = {}, purity = {}".format(to_print, ll, pur))
        return ll

    def train(self):
        """
        Conducts training

        inputs:
                None

        outputs:
                losses - list, list of losses obtained during training
                purities - list of [loss, purity, cluster_partition]
                cluster_part - the last cluster partition
                all_stats - all_stats on every EM-algorithm epoch
        """
        self.start_time = time.time()

        # preparing output templates
        losses = []
        purities = []
        cluster_part = 0
        all_stats = []

        # iterations over EM-algorithm epochs
        for epoch in range(self.max_epoch):
            if self.verbose:
                print("Beginning e-step")
            # preparing big_batch if needed
            if self.max_computing_size is not None:
                ids = np.random.permutation(self.N)[: self.max_computing_size]
                big_batch = self.X[ids].to(self.device)
            else:
                ids = None
                big_batch = None

            # E-step
            self.e_step(ids=ids)

            # Random model results
            if epoch == 0:
                if (ids is None) or (not self.full_purity):
                    clusters = torch.argmax(self.gamma, dim=0)
                else:
                    clusters = torch.argmax(
                        self.compute_gamma(
                            self.model(self.X.to(self.device)),
                            x=self.X,
                            size=(self.n_clusters, self.N),
                        ),
                        dim=0,
                    )
                if self.verbose:
                    print("Cluster partition")
                    for i in np.unique(clusters.cpu()):
                        print(
                            "Cluster",
                            i,
                            ": ",
                            np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                            " with pi = ",
                            self.pi[i],
                        )
                if self.target is not None:
                    random_pur = purity(
                        clusters,
                        self.target[ids]
                        if (ids is not None) and (not self.full_purity)
                        else self.target,
                    )
                else:
                    random_pur = None
                if self.verbose:
                    print("Purity for random model: {}".format(random_pur))

                # saving stats
                all_stats.append(dict())
                all_stats[-1]["gamma"] = self.get_gamma_stats()
                all_stats[-1]["model"] = self.get_model_stats()
                if big_batch is not None:
                    lambdas = self.model(big_batch)
                else:
                    lambdas = self.model(self.X)
                all_stats[-1]["lambdas"] = self.get_lambda_stats(lambdas)

            # M-step
            if self.verbose:
                print("Beginning m-step")
                for param_group in self.optimizer.param_groups:
                    lr = param_group["lr"]
                    break
                print("lr =", lr)
                print("lr_update_param =", self.lr_update_param)
            ll, ll_pur, cluster_part = self.m_step(big_batch=big_batch, ids=ids)
            
            # optuna part - report purity
            if self.trial:
                self.trial.report(ll_pur[1], epoch)

            # Handle pruning based on the intermediate value.
            if self.trial:
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # failure check
            if ll is None:
                return None, None, None, None

            # saving results
            losses += ll
            t = time.time()
            time_from_start = t - self.start_time
            purities.append(
                ll_pur[:2] + [cluster_part, self.n_clusters, time_from_start]
            )
            if self.verbose:
                print(
                    "On epoch {}/{} loss = {}, purity = {}, info = {}".format(
                        epoch + 1, self.max_epoch, ll_pur[0], ll_pur[1], ll_pur[2]
                    )
                )
                print("Time from start = {}".format(time_from_start))

            # saving model
            if self.best_model_path and (
                ll_pur[0] < self.prev_loss_model or epoch == 0
            ):
                if self.verbose:
                    print("Saving model")
                torch.save(self.model, self.best_model_path)
                self.prev_loss_model = ll_pur[0]

            # computing stats
            self.model.eval()
            with torch.no_grad():
                all_stats.append(dict())
                all_stats[-1]["gamma"] = self.get_gamma_stats()
                all_stats[-1]["model"] = self.get_model_stats()
                if big_batch is not None:
                    lambdas = self.model(big_batch)
                else:
                    lambdas = self.model(self.X)
                all_stats[-1]["lambdas"] = self.get_lambda_stats(lambdas)

            if (
                epoch > self.random_walking_max_epoch
                and self.n_clusters > self.true_clusters
            ):
                enforce = True
            else:
                enforce = False
                # updating number of clusters
            if epoch <= self.random_walking_max_epoch or enforce:
                if (
                    ((torch.rand(1) > 0.5)[0] or self.n_clusters == 1)
                    and self.n_clusters < self.upper_bound_clusters
                    and not enforce
                ):
                    split = True
                else:
                    split = False
                torch.save(self.model, os.path.join(self.exper_path, "tmp.pt"))
                pre_ll = float(self.compute_ll(big_batch, ids, "Before:"))
                if split:
                    if self.verbose:
                        print("Splitting")
                    for cluster in range(self.n_clusters):
                        self.model.to("cpu")
                        self.model.eval()
                        with torch.no_grad():
                            self.model.split_cluster(cluster, "cpu")
                        self.n_clusters += 1
                        self.model.to(self.device)
                        self.pi = torch.ones(self.n_clusters) / self.n_clusters
                        post_ll = float(
                            self.compute_ll(
                                big_batch,
                                ids,
                                "After splitting {} cluster:".format(cluster),
                            )
                        )
                        remain_prob = min(
                            1, math.exp(min(-post_ll + pre_ll, math.log(math.e)))
                        )
                        if self.verbose:
                            print("Remain probability: {}".format(remain_prob))
                        if (torch.rand(1) > remain_prob)[0]:
                            if self.verbose:
                                print("Loading model")
                            self.model = torch.load(os.path.join(self.exper_path, "tmp.pt"))
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                        else:
                            enforce = False
                            break
                else:
                    if enforce:
                        best_loss_enf = 1e9
                    if (torch.rand(1) > 0.5)[0]:
                        merge = True
                    else:
                        merge = False
                    if merge and not enforce:
                        if self.verbose:
                            print("Merging")
                        cluster_0 = int(torch.randint(self.n_clusters, size=(1,))[0])
                        for cluster_1 in range(self.n_clusters):
                            if cluster_1 == cluster_0:
                                continue
                            self.model.to("cpu")
                            self.model.eval()
                            with torch.no_grad():
                                self.model.merge_clusters(cluster_0, cluster_1, "cpu")
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            self.model.to(self.device)
                            post_ll = float(
                                self.compute_ll(
                                    big_batch,
                                    ids,
                                    "After merging {} and {} clusters:".format(
                                        cluster_0, cluster_1
                                    ),
                                )
                            )
                            remain_prob = min(
                                1, math.exp(min(-post_ll + pre_ll, math.log(math.e)))
                            )
                            if self.verbose:
                                print("Remain probability: {}".format(remain_prob))
                            if (torch.rand(1) > remain_prob)[0]:
                                if self.verbose:
                                    print("Loading model")
                                self.model = torch.load(
                                    os.path.join(self.exper_path, "tmp.pt")
                                )
                                self.n_clusters += 1
                                self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            else:
                                break
                    else:
                        if self.verbose:
                            print("Deleting")
                        for cluster in range(self.n_clusters):
                            self.model.to("cpu")
                            self.model.eval()
                            with torch.no_grad():
                                self.model.delete_cluster(cluster, "cpu")
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            self.model.to(self.device)
                            post_ll = float(
                                self.compute_ll(
                                    big_batch,
                                    ids,
                                    "After deleting {} cluster:".format(cluster),
                                )
                            )
                            remain_prob = min(
                                1, math.exp(min(-post_ll + pre_ll, math.log(math.e)))
                            )
                            if self.verbose:
                                print("Remain probability: {}".format(remain_prob))
                            if (torch.rand(1) > remain_prob)[0]:
                                if enforce:
                                    if post_ll < best_loss_enf:
                                        if self.verbose:
                                            print("Saving enforced model")
                                        best_loss_enf = post_ll
                                        torch.save(
                                            self.model,
                                            os.path.join(
                                                self.exper_path, "best_tmp.pt"
                                            ),
                                        )
                                if self.verbose:
                                    print("Loading model")
                                self.model = torch.load(os.path.join(self.exper_path, "tmp.pt"))
                                self.n_clusters += 1
                                self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            else:
                                break
                if enforce:
                    self.model = torch.load(
                        os.path.join(self.exper_path, "best_tmp.pt")
                    )
                    self.n_clusters -= 1
                    self.pi = torch.ones(self.n_clusters) / self.n_clusters
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                if self.max_computing_size is None:
                    self.gamma = torch.zeros(self.n_clusters, self.N).to(self.device)
                else:
                    self.gamma = torch.zeros(
                        self.n_clusters, self.max_computing_size
                    ).to(self.device)

        return losses, purities, cluster_part, all_stats


class TrainerClusterwiseForNH:
    """
        Trainer for multiple point processes clustering
    """

    def __init__(self, model, optimizer, device, N, dataloader, full_dataloader, n_clusters, target=None,
                 epsilon=1e-8, max_epoch=50, max_m_step_epoch=50, weight_decay=1e-5, lr=1e-3, lr_update_tol=25,
                 lr_update_param=0.5, random_walking_max_epoch=40, sigma_0 =1, sigma_inf = 0.005, inf_epoch = 50, true_clusters=5, upper_bound_clusters=10,
                 min_lr=None, updated_lr=None, batch_size=150, verbose=False, best_model_path=None,
                 max_computing_size=None, full_purity=True, zero_lambdas_test_plot=True, to_convolve = True):
        """
            inputs:
                    model - torch.nn.Module, model to train
                    optimizer - optimizer used for training
                    device - device, that is used for training
                    data - torch.Tensor, size = (N, sequence length, number of classes + 1),
                           partitions of the point processes
                    n_clusters - int, initial number of different point processes
                    target - torch.Tensor, size = (N), true labels or None
                    epsilon - float, used for log-s regularization log(x) -> log(x + epsilon)
                    max_epoch - int, number of epochs of EM algorithm
                    max_m_step_epoch - float, number of epochs of neural net training on M-step
                    lr_update_tol - int, tolerance before updating learning rate
                    lr_update_param - float, learning rate multiplier
                    random_walking_max_epoch - int, number of epochs when random walking of the number of clusters
                                               is available
                    true_clusters - int, true number of clusters
                    upper_bound_clusters - int, upper bound of the number of clusters
                    min_lr - float - minimal lr value, when achieved lr is updated to updated_lr and update params set
                             to default
                    updated_lr - float, lr after achieving min_lr
                    batch_size - int, batch size during neural net training
                    verbose - bool, if True, provides info during training
                    best_model_path - str, where the best model according to loss should be saved or None
                    max_computing_size - int, if not None, then constraints gamma size (one EM-algorithm step)
                    fool_purity - bool, if True, purity is computed on all dataset

            parameters:
                    N - int, number of data points
                    model - torch.nn.Module, model to train
                    optimizer - optimizer used for training
                    device - device used for training
                    X - torch.Tensor, size = (N, sequence length, number of classes + 1),
                        partitions of the point processes
                    target - torch.Tensor, size = (N), true labels or None
                    n_clusters - int, number of different point processes
                    max_epoch - int, number of epochs of EM algorithm
                    lr_update_tol - int, tolerance before updating learning rate
                    update_checker - int, checker, that is compared to tolerance, increased by one every time loss is
                                     greater then on the previous iteration
                    lr_update_param - float, learning rate multiplier
                    random_walking_max_epoch - int, number of epochs when random walking of the number of clusters
                                               is available
                    true_clusters - int, true number of clusters
                    upper_bound_clusters - int, upper bound of the number of clusters
                    min_lr - float - minimal lr value, when achieved lr is updated to updated_lr and update params set
                             to default
                    updated_lr - float, lr after achieving min_lr
                    epsilon - float, used for log-s regularization log(x) -> log(x + epsilon)
                    prev_loss - float, loss on previous iteration, used for updating update_checker
                    batch_size - int, batch size during neural net training
                    pi - torch.Tensor, size = (n_clusters), mixing coefficients, here are fixed and equal 1/n_clusters
                    gamma - torch.Tensor, size = (n_clusters, number of data points), probabilities p(k|x_n)
                    best_model_path - str, where the best model according to loss should be saved or None
                    prev_loss_model - float, loss obtained for the best model
                    max_computing_size - int, if not None, then constraints gamma size (one EM-algorithm step)
                    fool_purity - bool, if True, purity is computed on all dataset
        """
        self.N = N
        self.dataloader = dataloader
        self.full_dataloader = full_dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        if type(target):
            self.target = target.to(device)
        else:
            self.target = None
        self.n_clusters = n_clusters
        self.max_epoch = max_epoch
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_update_tol = lr_update_tol
        self.min_lr = min_lr
        self.updated_lr = updated_lr
        self.update_checker = -1
        self.epsilon = epsilon
        self.lr_update_param = lr_update_param
        self.prev_loss = 0
        self.max_m_step_epoch = max_m_step_epoch
        self.batch_size = batch_size
        self.pi = (torch.ones(n_clusters) / n_clusters).to(device)
        if max_computing_size is None:
            self.gamma = torch.zeros(n_clusters, self.N).to(device)
        else:
            self.gamma = torch.zeros(n_clusters, max_computing_size).to(device)
        self.max_computing_size = max_computing_size
        self.verbose = verbose
        self.best_model_path = best_model_path
        self.prev_loss_model = 0
        self.full_purity = full_purity
        self.start_time = time.time()
        self.random_walking_max_epoch = random_walking_max_epoch
        self.true_clusters = true_clusters
        self.upper_bound_clusters = upper_bound_clusters
        self.zero_lambdas_test_plot = zero_lambdas_test_plot
        self.to_convolve = to_convolve
        self.sigma = sigma_0
        self.tau = -1 / inf_epoch * math.log(sigma_inf / sigma_0)

    def compute_gamma(self, likelihood):
        """
            Computes gamma

            inputs:
                    lambdas - torch.Tensor, size = (batch_size or N, seq_len, number of classes), model output
                    x - torch.Tensor, size = (batch_size or N, seq_len, number of classes + 1),
                        data, that was processed or None
                    size - tuple, gamma size or None
                    device - device to compute

            outputs:
                    gamma - torch.Tensor, size = (n_clusters, batch_size or N), probabilities p(k|x_n)
        """
        tmp = torch.vstack(likelihood)
        tmp1 = torch.zeros_like(tmp)
        for i in range(self.n_clusters):
            tmp1[i,:] = 1/torch.sum(torch.exp(tmp - tmp[i,:][None,:]), dim = 0)
        return tmp1

    def convolve(self, gamma):
        """
            Convolves gamma along axis 0 with gaussian(0,sigma)

            inputs:
                    gamma - torch.Tensor, size = (n_clusters, batch_size), probabilities p(k|x_n)
            outputs:
                    convoluted - torch.Tensor, size = (n_clusters, batch_size), convoluted gamma and gauss
        """
        # initializing gauss
        gauss = torch.arange(- self.n_clusters + 1, self.n_clusters).float().to(self.device)
        gauss = torch.exp(-gauss ** 2 / (2 * self.sigma ** 2))
        gauss = gauss[:, None]

        # preparing output template
        convoluted = torch.zeros_like(gamma).to(self.device)

        # iterations over clusters
        for k in range(self.n_clusters):
            convoluted[k, :] = torch.sum(
                gauss[self.n_clusters - k - 1:2 * self.n_clusters - k - 1, :] * gamma.to(self.device), dim=0)

        # normalization
        convoluted /= convoluted.sum(dim=0)

        return convoluted

    def e_step(self, ids=None):
        """
            Conducts E-step of EM-algorithms, saves the result to self.gamma

            inputs:
                    None

            outputs:
                    None
        """
        self.model.eval()
        with torch.no_grad():
            for i, sampled_batch in enumerate(self.full_dataloader):
                event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length, ids = sampled_batch
                event_seqs_tensor = event_seqs_tensor.to(self.device)
                time_seqs_tensor = time_seqs_tensor.to(self.device)
                sim_time_seqs, sim_index_seqs = generate_sim_time_seqs(time_seqs_tensor.to('cpu'), seqs_length)
                sim_time_seqs = sim_time_seqs.to(self.device)
                self.model(event_seqs_tensor, time_seqs_tensor)
                likelihood = self.model.log_likelihood(event_seqs_tensor, sim_time_seqs, sim_index_seqs, last_time_seqs, seqs_length)
                self.gamma = self.compute_gamma(likelihood)
                break

    def train_epoch(self, big_batch=None):
        """
            Conducts one epoch of Neural Net training

            inputs:
                    None

            outputs:
                    log_likelihood - list of losses obtained during iterations over minibatches
        """
        # setting model to training and preparing output template
        self.model.train()
        log_likelihood = []

        # iterations over minibatches
        for i, sampled_batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length, ids = sampled_batch
            event_seqs_tensor = event_seqs_tensor.to(self.device)
            time_seqs_tensor = time_seqs_tensor.to(self.device)
            sim_time_seqs, sim_index_seqs = generate_sim_time_seqs(time_seqs_tensor.to('cpu'), seqs_length)
            sim_time_seqs = sim_time_seqs.to(self.device)
            self.model(event_seqs_tensor, time_seqs_tensor)
            nll = -torch.vstack(self.model.log_likelihood(event_seqs_tensor, sim_time_seqs, sim_index_seqs, last_time_seqs, seqs_length))
            if self.to_convolve:
                loss = torch.sum(self.convolve(self.gamma[:, ids]).cpu() * nll)
            else:
                loss = torch.sum(self.gamma[:, ids] * nll)
            loss.backward()
            self.optimizer.step()
            log_likelihood.append(loss.item())

        if np.mean(log_likelihood) > self.prev_loss:
            self.update_checker += 1
            if self.update_checker >= self.lr_update_tol:
                self.update_checker = 0
                lr = 0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_update_param
                    lr = param_group['lr']
                    if self.min_lr is not None:
                        if lr < self.min_lr:
                            param_group['lr'] = self.updated_lr
                if self.min_lr is not None:
                    if lr < self.min_lr:
                        lr = self.updated_lr
                self.lr = lr

        # saving previous loss
        self.prev_loss = np.mean(log_likelihood)

        return log_likelihood

    def m_step(self,  big_batch=None, ids=None):
        """
            Conducts M-step of EM-algorithm

            inputs:
                    None

            outputs:
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
        for epoch in range(int(self.max_m_step_epoch)):
            # one epoch training
            ll = self.train_epoch()
            log_likelihood_curve += ll

            # checking for failure
            if np.mean(ll) != np.mean(ll):
                return None, None, None

            # logs
            if epoch % 10 == 0 and self.verbose:
                print('Loss on sub_epoch {}/{}: {}'.format(epoch + 1,
                                                           self.max_m_step_epoch,
                                                           np.mean(ll)))

        # evaluating model
        self.model.eval()
        with torch.no_grad():
            for i, sampled_batch in enumerate(self.full_dataloader):
                event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length, ids = sampled_batch
                event_seqs_tensor = event_seqs_tensor.to(self.device)
                time_seqs_tensor = time_seqs_tensor.to(self.device)
                sim_time_seqs, sim_index_seqs = generate_sim_time_seqs(time_seqs_tensor.to('cpu'), seqs_length)
                sim_time_seqs = sim_time_seqs.to(self.device)
                self.model(event_seqs_tensor, time_seqs_tensor)
                likelihood = self.model.log_likelihood(event_seqs_tensor, sim_time_seqs, sim_index_seqs, last_time_seqs, seqs_length)
                gamma = self.compute_gamma(likelihood)
                break
            loss = torch.sum(-torch.vstack(likelihood)*gamma)
            clusters = torch.argmax(gamma, dim=0)
            if self.verbose:
                print('Cluster partition')
            cluster_partition = 2
            for i in np.unique(clusters.cpu()):
                if self.verbose:
                    print('Cluster', i, ': ', np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                          ' with pi = ', self.pi[i])
                cluster_partition = min(cluster_partition, np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters))
            if type(self.target):
                pur = purity(clusters.to('cpu'), self.target.to('cpu')[ids])
                info = info_score(clusters.to('cpu'), self.target.to('cpu'), len(np.unique(self.target.to('cpu'))))
            else:
                pur = -1
                info = -1
            if self.zero_lambdas_test_plot:
                event_seqs_tensor = event_seqs_tensor[0]
                time_seqs_tensor = time_seqs_tensor[0]
                time_cum_seqs_tensor = torch.cumsum(time_seqs_tensor, dim = 0)
                times = np.linspace(0, float(time_cum_seqs_tensor[-1].cpu()), 100)
                lambdas = self.model.get_lambdas(event_seqs_tensor, time_seqs_tensor, time_cum_seqs_tensor, times, self.device).detach().cpu().numpy()
                for k in range(self.n_clusters):
                    plt.plot(times, lambdas[k])
                    plt.scatter(time_cum_seqs_tensor.detach().cpu(), torch.ones(len(time_cum_seqs_tensor.detach().cpu())))
                    plt.show()

        return log_likelihood_curve, [loss, pur, info], cluster_partition

    def compute_ll(self,big_batch, idx, to_print):
        with torch.no_grad():
            for i, sampled_batch in enumerate(self.full_dataloader):
                event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length, ids= sampled_batch
                event_seqs_tensor = event_seqs_tensor.to(self.device)
                time_seqs_tensor = time_seqs_tensor.to(self.device)
                sim_time_seqs, sim_index_seqs = generate_sim_time_seqs(time_seqs_tensor.to('cpu'), seqs_length)
                sim_time_seqs = sim_time_seqs.to(self.device)
                self.model(event_seqs_tensor, time_seqs_tensor)
                likelihood = self.model.log_likelihood(event_seqs_tensor, sim_time_seqs, sim_index_seqs, last_time_seqs, seqs_length)
                gamma = self.compute_gamma(likelihood)
                break
            clusters = torch.argmax(gamma, dim=0)
        ll = -torch.sum(torch.vstack(likelihood))
        if self.verbose:
            print('Cluster partition')
        cluster_partition = 2
        for i in np.unique(clusters.cpu()):
            if self.verbose:
                print('Cluster', i, ': ', np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                      ' with pi = ', self.pi[i])
            cluster_partition = min(cluster_partition, np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters))
        if type(self.target):
            pur = purity(clusters.to('cpu'), self.target.to('cpu')[ids])
        else:
            pur = None
        if self.verbose:
            print('{} loss = {}, purity = {}'.format(to_print, ll, pur))
        return ll

    def train(self):
        """
            Conducts training

            inputs:
                    None

            outputs:
                    losses - list, list of losses obtained during training
                    purities - list of [loss, purity, cluster_partition]
                    cluster_part - the last cluster partition
                    all_stats - all_stats on every EM-algorithm epoch
        """
        self.start_time = time.time()

        # preparing output templates
        losses = []
        purities = []
        cluster_part = 0
        all_stats = []

        # iterations over EM-algorithm epochs
        for epoch in range(self.max_epoch):
            if self.verbose:
                print('Beginning e-step')
            # preparing big_batch if needed
            if self.max_computing_size is not None:
                ids = np.random.permutation(self.N)[:self.max_computing_size]
                big_batch = self.X[ids].to(self.device)
            else:
                ids = None
                big_batch = None

            # E-step
            self.e_step(ids=ids)

            # Random model results
            if epoch == 0:
                if (ids is None) or (not self.full_purity):
                    clusters = torch.argmax(self.gamma, dim=0)
                else:
                    raise Exception('Not Implemented')
                    #clusters = torch.argmax(self.compute_gamma(self.model(self.X.to(self.device)), x=self.X,
                    #                                           size=(self.n_clusters, self.N)), dim=0)
                if self.verbose:
                    print('Cluster partition')
                    for i in np.unique(clusters.cpu()):
                        print('Cluster', i, ': ', np.sum((clusters.cpu() == i).cpu().numpy()) / len(clusters),
                              ' with pi = ', self.pi[i])
                if type(self.target):
                    random_pur = purity(clusters.cpu(), self.target.cpu())
                else:
                    random_pur = None
                if self.verbose:
                    print('Purity for random model: {}'.format(random_pur))

            # M-step
            if self.verbose:
                print('Beginning m-step')
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    break
                print('lr =', lr)
                print('lr_update_param =', self.lr_update_param)
                print('sigma =', self.sigma)
            ll, ll_pur, cluster_part = self.m_step(big_batch=big_batch, ids=ids)
            self.sigma *= np.exp(-self.tau)
            # failure check
            if ll is None:
                return None, None, None, None

            # saving results
            losses += ll
            t = time.time()
            time_from_start = t - self.start_time
            purities.append(ll_pur[:2] + [cluster_part, self.n_clusters, time_from_start])
            if self.verbose:
                print('On epoch {}/{} loss = {}, purity = {}, info = {}'.format(epoch + 1, self.max_epoch,
                                                                                ll_pur[0], ll_pur[1], ll_pur[2]))
                print('Time from start = {}'.format(time_from_start))

            # saving model
            if self.best_model_path and (ll_pur[0] < self.prev_loss_model or epoch == 0):
                if self.verbose:
                    print('Saving model')
                torch.save(self.model, self.best_model_path)
                self.prev_loss_model = ll_pur[0]

            if epoch > self.random_walking_max_epoch and self.n_clusters > self.true_clusters:
                enforce = True
            else:
                enforce = False
            # updating number of clusters
            if epoch <= self.random_walking_max_epoch or enforce:
                if ((torch.rand(1) > 0.5)[
                        0] or self.n_clusters == 1) and self.n_clusters < self.upper_bound_clusters and not enforce:
                    split = True
                else:
                    split = False
                torch.save(self.model, 'tmp.pt')
                pre_ll = float(self.compute_ll(big_batch, ids, 'Before:'))
                if split:
                    if self.verbose:
                        print('Splitting')
                    for cluster in range(self.n_clusters):
                        self.model.to('cpu')
                        self.model.eval()
                        with torch.no_grad():
                            self.model.split_cluster(cluster, 'cpu')
                        self.n_clusters += 1
                        self.model.to(self.device)
                        self.pi = torch.ones(self.n_clusters) / self.n_clusters
                        post_ll = float(self.compute_ll(big_batch, ids, 'After splitting {} cluster:'.format(cluster)))
                        remain_prob = min(1, math.exp(min(- post_ll + pre_ll, math.log(math.e))))
                        if self.verbose:
                            print('Remain probability: {}'.format(remain_prob))
                        if (torch.rand(1) > remain_prob)[0]:
                            if self.verbose:
                                print('Loading model')
                            self.model = torch.load('tmp.pt')
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                        else:
                            enforce = False
                            break
                else:
                    if enforce:
                        best_loss_enf = 1e+9
                    if (torch.rand(1) > 0.5)[0]:
                        merge = True
                    else:
                        merge = False
                    if merge and not enforce:
                        if self.verbose:
                            print('Merging')
                        cluster_0 = int(torch.randint(self.n_clusters, size=(1,))[0])
                        for cluster_1 in range(self.n_clusters):
                            if cluster_1 == cluster_0:
                                continue
                            self.model.to('cpu')
                            self.model.eval()
                            with torch.no_grad():
                                self.model.merge_clusters(cluster_0, cluster_1, 'cpu')
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            self.model.to(self.device)
                            post_ll = float(self.compute_ll(big_batch, ids,
                                                            'After merging {} and {} clusters:'.format(cluster_0,
                                                                                                       cluster_1)))
                            remain_prob = min(1, math.exp(min(- post_ll + pre_ll, math.log(math.e))))
                            if self.verbose:
                                print('Remain probability: {}'.format(remain_prob))
                            if (torch.rand(1) > remain_prob)[0]:
                                if self.verbose:
                                    print('Loading model')
                                self.model = torch.load('tmp.pt')
                                self.n_clusters += 1
                                self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            else:
                                break
                    else:
                        if self.verbose:
                            print('Deleting')
                        for cluster in range(self.n_clusters):
                            self.model.to('cpu')
                            self.model.eval()
                            with torch.no_grad():
                                self.model.delete_cluster(cluster, 'cpu')
                            self.n_clusters -= 1
                            self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            self.model.to(self.device)
                            post_ll = float(
                                self.compute_ll(big_batch, ids, 'After deleting {} cluster:'.format(cluster)))
                            remain_prob = min(1, math.exp(min(- post_ll + pre_ll, math.log(math.e))))
                            if self.verbose:
                                print('Remain probability: {}'.format(remain_prob))
                            if (torch.rand(1) > remain_prob)[0]:
                                if enforce:
                                    if post_ll < best_loss_enf:
                                        if self.verbose:
                                            print('Saving enforced model')
                                        best_loss_enf = post_ll
                                        torch.save(self.model, 'best_tmp.pt')
                                if self.verbose:
                                    print('Loading model')
                                self.model = torch.load('tmp.pt')
                                self.n_clusters += 1
                                self.pi = torch.ones(self.n_clusters) / self.n_clusters
                            else:
                                break
                if enforce:
                    self.model = torch.load('best_tmp.pt')
                    self.n_clusters -= 1
                    self.pi = torch.ones(self.n_clusters) / self.n_clusters
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                if self.max_computing_size is None:
                    self.gamma = torch.zeros(self.n_clusters, self.N).to(self.device)
                else:
                    self.gamma = torch.zeros(self.n_clusters, self.max_computing_size).to(self.device)

        return losses, purities, cluster_part, all_stats