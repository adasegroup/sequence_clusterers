__all__ = ['RandomGeneratedSequences',
           'RandomGeneratedSequencesWithIndex']

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.datamodule import get_partition


class RandomGeneratedSequences(Dataset):
    """
    This Dataset allows to work with different types of data in the same format:
    - real world dataset like AgeClusterData or LinkedIn or IPTV dasets, which consist real sequences of different events
    - generated data
    There are two types of generated random data based on Hawkes process

    """
    def __init__(self, path_to_csvs, num_of_event_types,
                 num_of_steps, predefined_step_size=False):
        self.path = path_to_csvs
        files = list(filter(lambda x: x != 'clusters.csv', os.listdir(self.path)))
        data_mask = [file.endswith('.csv') for file in files]
        self.targets = torch.from_numpy(np.array(pd.read_csv(self.path + '/clusters.csv')['cluster_id']))
        self.file_paths = np.array(files)[data_mask].tolist()
        self.num_of_steps = num_of_steps
        self.num_of_event_types = num_of_event_types + 1
        self.predefined_step_size = predefined_step_size
        self.partitions = torch.zeros(len(self.file_paths), num_of_steps, self.num_of_event_types)
        for i, file_path in enumerate(self.file_paths):
            sample = np.array(pd.read_csv(os.path.join(self.path, self.file_paths[i])))
            sample_df = pd.DataFrame(sample, columns=['idx', 'time', 'event'])
            sample_df = sample_df.drop(columns='idx')
            partitions = get_partition(sample_df, self.num_of_steps, self.num_of_event_types - 1)
            self.partitions[i] = partitions

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.partitions[index], self.targets[index]


class RandomGeneratedSequencesWithIndex(RandomGeneratedSequences):
    def __getitem__(self, index):
        return self.partitions[index], self.targets[index], index
