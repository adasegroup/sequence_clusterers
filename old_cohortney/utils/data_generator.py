"""
   This file contains function generate_dataset for data generation based on the model from models/LSTM.py
"""

# setting parent directory
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# imports
from models.LSTM import LSTMSinglePointProcess
import torch


def generate_dataset(n_classes, n_clusters, n_points_per_cluster, n_timestamps,
                     hidden_size, num_layers, dt_scaling_param):
    """
        input:
               n_classes - int, number of event types, that can occur
               n_clusters - int, number of different point processes to generate
               n_points_per_clusters - int, number of runs for each point process
               n_timestamps - int, sequence length
               hidden_size - int, LSTM hidden size
               num_layers - int, number of layers in LSTM
               dt_scaling_param - float, used for random delta_time generation, dt = torch.rand()*dt_scaling_param

        output:
               data - torch.Tensor, size = (n_clusters*n_points_per_cluster, n_timestamps, n_classes + 1),
                      generated data
               target - torch.Tensor, size = (n_clusters*n_points_per_cluster), true cluster assignment
    """
    data = torch.zeros(n_clusters * n_points_per_cluster, n_timestamps, n_classes + 1)
    target = torch.zeros(n_clusters * n_points_per_cluster)
    for k in range(n_clusters):
        model = LSTMSinglePointProcess(n_classes + 1, hidden_size, num_layers, n_classes)
        data[k * n_points_per_cluster:(k + 1) * n_points_per_cluster, :, :] \
            = model.simulate(n_points_per_cluster, torch.rand(n_points_per_cluster) * dt_scaling_param,
                             n_timestamps, True)
        target[k * n_points_per_cluster:(k + 1) * n_points_per_cluster] = k
    return data, target
