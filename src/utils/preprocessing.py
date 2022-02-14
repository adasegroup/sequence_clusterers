import os
import pandas as pd
import torch
import tqdm
import numpy as np


def cmp_to_key(mycmp):
    # TODO get rid of it
    """
        Convert a cmp= function into a key= function
    """

    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def compare(a, b):
    # TODO change
    tmp1 = int(a[:-4])
    tmp2 = int(b[:-4])
    return tmp1 - tmp2


def get_partition(df, num_of_steps, num_of_classes, end_time=None):
    # TODO optimize
    """
        Transforms dataset into partition

        inputs:
                df - pandas.DataFrame, columns - time and event
                num_of_steps - int, number of steps in partition
                num_of_classes - int, number of event types
                end_time - float, end time or None

        outputs:
                partition - torch.Tensor, size = (num_of_steps, num_of_classes + 1)
    """
    df = df.loc[:, ['time', 'event']].copy()
    df = df.sort_values(by=['time'])
    df = df.reset_index().loc[:, ['time', 'event']].copy()
    # setting end time if None
    if end_time is None:
        end_time = df['time'][len(df['time']) - 1]

    # preparing output template
    res = torch.zeros(num_of_steps, num_of_classes + 1)

    # finding delta time
    dt = end_time / num_of_steps
    res[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    df['time'] = (df['time'] / dt).astype(int)
    mask = (df['time'] == num_of_steps)
    df.loc[mask, 'time'] -= 1

    # counting points
    df = df.reset_index()
    df = df.groupby(['time', 'event']).count()
    df = df.reset_index()
    df.columns = ['time', 'event', 'num']
    df['time'] -= df['time'].iloc[0]
    try:
        df['event'] = df['event'].astype(int)
    except:
        global evnts
        global cur
        for i in range(len(df['event'])):
            if df['event'].iloc[i] not in evnts:
                evnts[df['event'].iloc[i]] = cur
                cur += 1
            df['event'].iloc[i] = evnts[df['event'].iloc[i]]
        df['event'] = df['event'].astype(int)

    # computing partition
    tmp = torch.Tensor(df.to_numpy()).long()
    res[tmp[:, 0], tmp[:, 1] + 1] = tmp[:, 2].float()

    return res


def get_dataset(path_to_files, n_classes, n_steps, n_files = None):
    """
        Reads dataset

        inputs:
                path_to_files - str, path to csv files with dataset
                n_classes - int, number of event types
                n_steps - int, number of steps in partitions

        outputs:
                data - torch.Tensor, size = (N, n_steps, n_classes + 1), dataset
                target - torch.Tensor, size = (N), true labels or None
    """
    # searching for files
    files = os.listdir(path_to_files)
    target = None
    last_event_target = False
    
    if n_files is not None:
        N =  len(files)
        if 'clusters.csv' in files:
            N -= 1
        if 'info.json' in files:
            N -= 1
        ids = np.random.permutation(N)[:n_files]
        ids = np.sort(ids)

    # reading target
    if 'clusters.csv' in files:
        files.remove('clusters.csv')
        target = torch.Tensor(pd.read_csv(path_to_files + '/clusters.csv')['cluster_id'])
        if n_files is not None:
            target = target[ids]
    if 'info.json' in files:
        files.remove('info.json')

    # reading data
    files = sorted(files, key=cmp_to_key(compare))
    if n_files is not None:
        files = [files[i] for i in ids]
    data = torch.zeros(len(files), n_steps, n_classes + 1)
    for i, f in tqdm.tqdm(enumerate(files)):
        df = pd.read_csv(path_to_files + '/' + f)
        data[i, :, :] = get_partition(df, n_steps, n_classes)
    
    data[:,:,0] = (data[:,:,0] - torch.min(data[:,:,0]))/(torch.max(data[:,:,0]) - torch.min(data[:,:,0]))
    
    return data, target


def preprocess(data_dir, preprocessing: dict):
    if preprocessing['type'] == 'equipartition':
        data, target = get_dataset(data_dir, preprocessing['n_classes'], preprocessing['n_steps'], preprocessing['n_files'] if 'n_files' in preprocessing else None)
        return data, target
    else:
        raise Exception('Unknown preprocessing type')