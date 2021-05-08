# +
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
        sample = torch.tensor(np.array(pd.read_csv(self.path + data[index])))
        return sample, target
    
    
def get_partition(sample, num_of_steps, num_of_event_types, end_time=None):
    '''
        Transforms a sample into partition
        inputs:
                sample -  pd.DataFrame; columns - time and type of event, sorted by time
                num_of_steps - int, number of steps in partition
                num_of_event_type - int, number of event types
                end_time - float, end time or None
        outputs:
                partition - torch.tensor, size = (num_of_steps, num_of_classes + 1)
    '''
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
    temp = torch.Tensor(sample.to_numpy()).long()
    partition[temp[:, 0], temp[:, 1] + 1] = temp[:, 2].float()
    return partition



def batch_to_partitions(batch, num_of_steps, num_of_event_types, end_time=None):
    '''
    The function produces partitions for batch of data of RGS dataset
    '''
    partitions = torch.zeros(len(batch), num_of_steps, num_of_event_types + 1)
    for num_of_sample, sample in enumerate(batch):
        sample_df = pd.DataFrame(sample.numpy(), columns=['idx', 'time', 'event'])
        sample_df = sample_df.drop(columns='idx')
        partitions[num_of_sample] = get_partition(sample_df, num_of_steps, num_of_event_types)
    return partitions
