import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RandomGeneratedSequences(Dataset):
    def __init__(self, path_to_csvs, num_of_event_types,
                 num_of_steps=-1, predefined_step_size=False):
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


def get_partition(sample, num_of_steps, num_of_event_types, end_time=None):
    """
    Transforms a sample into partition
    inputs:
            sample -  pd.DataFrame; columns - time and type of event, sorted by time
            num_of_steps - int, number of steps in partition
            num_of_event_type - int, number of event types
            end_time - float, end time or None
    outputs:
            partition - torch.tensor, size = (num_of_steps, num_of_classes + 1)
    """
    if end_time is None:
        end_time = sample['time'][len(sample['time']) - 1]

    partition = torch.zeros(num_of_steps, num_of_event_types + 1)

    # finding time stamp
    dt = end_time / num_of_steps
    partition[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    sample['time'] = (sample['time'] / dt).astype(int)
    mask = (sample['time'] == num_of_steps)
    sample.loc[mask, 'time'] -= 1

    # counting points
    sample = sample.reset_index()
    sample = sample.groupby(['time', 'event']).count()
    sample = sample.reset_index()
    sample.columns = ['time', 'event', 'num']
    try:
        sample['event'] = sample['event'].astype(int)
    except:
        global events
        global cur
        for i in range(len(sample['event'])):
            if sample['event'].iloc[i] not in events:
                events[sample['event'].iloc[i]] = cur
                cur += 1
            sample['event'].iloc[i] = events[sample['event'].iloc[i]]
        sample['event'] = sample['event'].astype(int)

    # computing partition
    temp = torch.from_numpy(sample.to_numpy())
    partition[temp[:, 0], temp[:, 1] + 1] = temp[:, 2].float()
    return partition


if __name__ == '__main__':
    dataset_ex = RandomGeneratedSequences('data/sin_K2_C5', num_of_steps=1000, num_of_event_types=8)
    partitions = []
