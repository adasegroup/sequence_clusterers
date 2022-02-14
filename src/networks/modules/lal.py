"""
    This file contains LSTM based models for intensity prediction
"""

# imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from typing import Tuple


# modules
class ScaledSoftplus(nn.Module):
    """
    Scaled softplus model, returns f(x) = s * log(1 + exp(x/s))
    """

    def __init__(self):
        super().__init__()
        self.s = Parameter(torch.ones(1))

    def forward(self, x):
        return self.s * torch.log(1 + torch.exp(x / self.s))


class LSTMMultiplePointProcesses(nn.Module):
    """
    Multiple Point Processes Model, Point Processes are distinguished with different initial hidden states
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        num_clusters: int,
        upper_bound_clusters: int,
        n_steps: int,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        input_size = num_classes + 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_steps = n_steps
        self.batch_first = batch_first
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.norm = nn.BatchNorm1d(n_steps)
        if bidirectional:
            self.hidden0 = Parameter(
                self.init_weigh(upper_bound_clusters, num_layers * 2, hidden_size)
            )
            self.cell0 = Parameter(
                self.init_weigh(upper_bound_clusters, num_layers * 2, hidden_size)
            )
            self.W = Parameter(self.init_weigh(hidden_size * 2, num_classes))
        else:
            self.hidden0 = Parameter(
                self.init_weigh(upper_bound_clusters, num_layers, hidden_size)
            )
            self.cell0 = Parameter(
                self.init_weigh(upper_bound_clusters, num_layers, hidden_size)
            )
            self.W = Parameter(self.init_weigh(hidden_size, num_classes))

        self.hidden0_reserved = self.hidden0.data.detach().clone()
        self.cell0_reserved = self.cell0.data.detach().clone()
        self.s_reserved = [torch.ones(1)] * upper_bound_clusters

        for k in range(upper_bound_clusters):
            setattr(self, "f_{}".format(k), ScaledSoftplus())
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.prev_num_clusters = num_clusters
        self.upper_bound_clusters = upper_bound_clusters
        self.bidir = bidirectional

    def get_params(self):
        return {
            "name": "LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "upper_bound_clusters": self.upper_bound_clusters,
            "num_clusters": self.num_clusters,
            "n_steps": self.n_steps,
            "batch_first": self.batch_first,
            "dropout": self.dropout,
            "bidirectional": self.bidir,
        }

    def init_weigh(self, *args):
        """
        Used for weight initialization, output ~ U(-1/hidden_size,1/hidden_size)
        """
        tmp = self.hidden_size
        return torch.rand(*args) * 2 / tmp - 1 / tmp

    def reserve_weighs(self):
        self.prev_num_clusters = self.num_clusters
        self.hidden0_reserved = self.hidden0.data.detach().clone()
        self.cell0_reserved = self.cell0.data.detach().clone()
        self.s_reserved = [
            getattr(self, "f_{}".format(k)).s.data.detach().clone()
            for k in range(self.upper_bound_clusters)
        ]

    def reverse_weighs(self):
        self.num_clusters = self.prev_num_clusters
        self.hidden0.data = self.hidden0_reserved.detach().clone()
        self.cell0.data = self.cell0_reserved.detach().clone()
        for k in range(self.upper_bound_clusters):
            getattr(self, "f_{}".format(k)).s.data = self.s_reserved[k].detach().clone()

    def merge_clusters(self, cluster_0, cluster_1):
        """
        Used for merging two clusters
        """
        self.reserve_weighs()

        # checking if merging is allowed
        assert self.num_clusters > 1
        assert cluster_0 != cluster_1

        if cluster_0 > cluster_1:
            cluster_0, cluster_1 = cluster_1, cluster_0

        # merging
        self.hidden0.data[cluster_0, ...] = (
            (
                (self.hidden0.data[cluster_0, ...] + self.hidden0.data[cluster_1, ...])
                / 2
            )
            .detach()
            .clone()
        )
        self.hidden0.data[cluster_1:-1, ...] = (
            self.hidden0.data[cluster_1 + 1 :, ...].detach().clone()
        )
        self.cell0.data[cluster_0, ...] = (
            ((self.cell0.data[cluster_0, ...] + self.cell0.data[cluster_1, ...]) / 2)
            .detach()
            .clone()
        )
        self.cell0.data[cluster_1:-1, ...] = (
            self.cell0.data[cluster_1 + 1 :, ...].detach().clone()
        )

        # updating number of clusters
        self.num_clusters -= 1

        for k in range(cluster_1, self.num_clusters):
            getattr(self, "f_{}".format(k)).s.data = getattr(
                self, "f_{}".format(k + 1)
            ).s.data

    def delete_cluster(self, cluster):
        """
        Used for deleting a cluster
        """
        self.reserve_weighs()

        # checking that deleting is important
        assert self.num_clusters > 1

        # deleting
        self.hidden0.data[cluster:-1, ...] = (
            self.hidden0.data[cluster + 1 :, ...].detach().clone()
        )
        self.cell0.data[cluster:-1, ...] = (
            self.cell0.data[cluster + 1 :, ...].detach().clone()
        )

        # updating number of clusters
        self.num_clusters -= 1

        for k in range(cluster, self.num_clusters):
            getattr(self, "f_{}".format(k)).s.data = (
                getattr(self, "f_{}".format(k + 1)).s.data.detach().clone()
            )

    def split_cluster(self, cluster):
        """
        Used for splitting a cluster
        """
        self.reserve_weighs()

        # splitting
        a = np.random.beta(1, 1)
        self.hidden0.data[cluster, ...], self.hidden0.data[self.num_clusters, ...] = (
            2 * a * self.hidden0.data[cluster, ...].detach().clone(),
            2 * (1 - a) * self.hidden0.data[cluster, ...].detach().clone(),
        )

        self.cell0.data[cluster, ...], self.cell0.data[self.num_clusters, ...] = (
            2 * a * self.cell0.data[cluster, ...].detach().clone(),
            2 * (1 - a) * self.cell0.data[cluster, ...].detach().clone(),
        )

        # updating number of clusters
        self.num_clusters += 1

        getattr(self, "f_{}".format(self.num_clusters - 1)).s.data = (
            getattr(self, "f_{}".format(cluster)).s.data.detach().clone()
        )

    def compute_gamma(
        self,
        lambdas: torch.Tensor,
        x: torch.Tensor,
        size: Tuple[int, int],  # shape = [n_clusters, batch_size]
        epsilon: float = 1e-8,
        device="cpu",
    ) -> torch.Tensor:
        """
        computes gamma
        """
        # preparing gamma template
        gamma = torch.zeros(size).to(device)

        # preparing delta times and partitions for computing gamma
        dts = x[:, 0, 0]
        dts = dts[None, :, None, None].to(device)
        partitions = x[:, :, 1:]
        partitions = partitions[None, :, :, :].to(device)

        # iterations over clusters
        for k in range(self.num_clusters):
            # lambdas of current cluster
            lambdas_k = lambdas[k, :, :, :]
            lambdas_k = lambdas_k[None, :, :, :].to(device)

            # computing gamma for k-th cluster
            tmp_sub = (lambdas - lambdas_k) * dts
            tmp = torch.sum(
                -tmp_sub
                + partitions
                * (torch.log(lambdas + epsilon) - torch.log(lambdas_k + epsilon)),
                dim=(2, 3),
            )
            tmp = 1 / (torch.sum(torch.exp(tmp), dim=0))  # shape = [batch_size,]

            # resolving nans
            tmp[tmp != tmp] = 0

            gamma[k, :] = tmp

        return gamma

    def forward(self, s, epsilon, device="cpu", return_states=False):
        """
        forward pass of the model
        """
        bs, seq_len, _ = s.shape

        # preparing lambdas template
        lambdas = torch.zeros(self.num_clusters, bs, seq_len, self.num_classes).to(
            device
        )

        # iterating over point processes
        hiddens = []
        cells = []
        for k in range(self.num_clusters):
            # hidden and cell state preprocessing
            hidden0 = self.hidden0[k, :, None, :].repeat(1, bs, 1)
            cell0 = self.cell0[k, :, None, :].repeat(1, bs, 1)

            # LSTM forward pass
            out, (hidden, cell) = self.lstm(s, (hidden0, cell0))
            hiddens.append(hidden)
            cells.append(cell)

            out = self.norm(out)

            # finding lambdas
            if self.bidir:
                h0 = hidden0[-3:, 0, :].reshape(-1)
            else:
                h0 = hidden0[-1, 0, :].reshape(-1)

            # first lambda doesn't depend on history and depends only on the initial hidden state
            lambdas[k, :, 0, :] = getattr(self, "f_{}".format(k))(h0 @ self.W)[
                None, :
            ].repeat(bs, 1)
            lambdas[k, :, 1:, :] = getattr(self, "f_{}".format(k))(
                out[:, :-1, :] @ self.W
            )

        gamma = self.compute_gamma(
            lambdas, s, (self.num_clusters, s.shape[0]), epsilon, device=device
        )

        if return_states:
            return lambdas, gamma, hiddens, cells
        return lambdas, gamma
