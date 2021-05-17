import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from models.models import LSTM_single_point_process
import torch
import torch.nn as nn

def generate_dataset(n_classes, n_clusters, n_points_per_cluster, n_timestamps,\
                     hidden_size,num_layers, dt_scaling_param, save_path, verbose = False):
    data = torch.zeros(n_clusters*n_points_per_cluster, n_timestamps, n_classes+1)
    target = torch.zeros(n_clusters*n_points_per_cluster)
    for k in range(n_clusters): 
        model = LSTM_single_point_process(n_classes+1, hidden_size, num_layers,n_classes)
        data[k*n_points_per_cluster:(k+1)*n_points_per_cluster,:,:]\
            = model.simulate(n_points_per_cluster, torch.rand(n_points_per_cluster)*dt_scaling_param,
                             n_timestamps, True)
        target[k*n_points_per_cluster:(k+1)*n_points_per_cluster] = k
    return data, target