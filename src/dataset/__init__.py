import torch
import os
import numpy as np
import pandas as pd

class RandomGeneratedSequences(torch.utils.data.Dataset):
    def __init__(self, path_to_csvs):
        self.path = path_to_csvs
        
    def __len__(self):
        files = os.listdir(self.path)
        if 'clusters.csv' in files:
            files.remove('clusters.csv')
        return sum([file.endswith('.csv') for file in files])
    
    def __getitem__(self, index):
        files = os.listdir(self.path)
        target = None #
        if 'clusters.csv' in files:
            target = torch.Tensor(pd.read_csv(self.path + '/clusters.csv')['cluster_id'])[index]
            files.remove('clusters.csv')
        data_mask = [file.endswith('.csv') for file in files]
        data = np.array(files)[data_mask].tolist()
        sample = torch.Tensor(np.array(pd.read_csv(self.path + data[index])))
        return sample, target