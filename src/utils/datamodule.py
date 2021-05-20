import os
import re
import zipfile
import shutil

import torch
from pathlib import Path
import numpy as np
import pandas as pd

from src.dataset import dataset_urls


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


def load_data(data_dir, maxsize=None, maxlen=-1, ext='txt', datetime=True, type_=None):
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    (str, Path) - directory containing sequences
        maxsize     (int)       - maximum number of sequences to load
        maxlen      (int)       - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         (str)       - extension of files in data_dir directory
        datetime    (bool)      - variable meaning if time values in files are represented in datetime format

    Returns:
        ss          (List(torch.Tensor))    - list of torch.Tensor containing sequences. Each tensor has shape (L, 2) and represents event sequence
                                                as sequence of pairs (t, c). t - time, c - event type.
        Ts          (torch.Tensor)          - tensor of right edges T_n of interavls (0, T_n) in which point processes realizations lie.
        class2idx   (Dict)                  - dict of event types and their indexes
        user_list   (List(Dict))            - representation of sequences siutable for Cohortny

    """

    s = []
    classes = set()
    nb_files = 0
    for file in sorted(os.listdir(data_dir),
                       key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0):
        if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
            if maxsize is None or nb_files <= maxsize:
                nb_files += 1
            else:
                break

            time_col = 'time'
            event_col = 'event'
            if type_ == 'booking1':
                time_col = 'checkin'
                event_col = 'city_id'
            elif type_ == 'booking2':
                time_col = 'checkout'
                event_col = 'city_id'

            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df[event_col].unique()))
            if datetime:
                df[time_col] = pd.to_datetime(df[time_col])
                df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1, 'D')
            if maxlen > 0:
                df = df.iloc[:maxlen]

            s.append(df)

    classes = list(classes)
    class2idx = {clas: idx for idx, clas in enumerate(classes)}

    ss, Ts = [], []
    user_list = []
    for i, df in enumerate(s):
        user_dict = dict()
        if s[i][time_col].to_numpy()[-1] < 0:
            continue
        s[i][event_col].replace(class2idx, inplace=True)
        for event_type in class2idx.values():
            dat = s[i][s[i][event_col] == event_type]
            user_dict[event_type] = dat[time_col].to_numpy()
        user_list.append(user_dict)

        st = np.vstack([s[i][time_col].to_numpy(), s[i][event_col].to_numpy()])
        tens = torch.FloatTensor(st.astype(np.float32)).T

        if maxlen > 0:
            tens = tens[:maxlen]
        ss.append(tens)
        Ts.append(tens[-1, 0])

    Ts = torch.FloatTensor(Ts)

    return ss, Ts, class2idx, user_list


def sep_hawkes_proc(user_list, event_type):
    """
    transforming data to the array taking into account an event type
    :param user_list:
    :param event_type:
    :return:
    """
    sep_seqs = []
    for user_dict in user_list:
        sep_seqs.append(np.array(user_dict[event_type], dtype=np.float32))

    return sep_seqs


def check_existance(dir) -> bool:
    return os.path.exists(dir)


def download_unpack_zip(zipurl: str, data_dir):
    res_path = '/'.join(data_dir.split('/')[:-1])
    download_link(zipurl, destination=res_path)
    zip_name = zipurl.split('/')[-1]
    unpack(lfilename=os.path.join(res_path, zip_name), dir=res_path)


def download_link(url, destination='data'):
    print(destination)
    res_code = os.system(f'wget {url} -P {destination}')
    if res_code != 0:
        raise Exception('Download data with some problem')


def unpack(lfilename, dir):
    with zipfile.ZipFile(lfilename) as file:
        os.makedirs(dir, exist_ok=True)
        file.extractall(dir)
    if check_existance(os.path.join(dir, '__MACOSX')):
        shutil.rmtree(os.path.join(dir, '__MACOSX'),)

    os.remove(lfilename)


def download_dataset(data_dir, data_name):
    """
    Download dataset is it is availible
    :return:
    """
    if not check_existance(data_dir):
        download_unpack_zip(dataset_urls[data_name], data_dir)

