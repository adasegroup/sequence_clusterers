import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
import tsfresh
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
    EfficientFCParameters,
)
from tslearn.utils import to_time_series_dataset

events = {}
cur = 0


def pad_time(instances, pad):
    """Pad the instance to the max seq length in batch."""
    max_len = max(len(inst) for inst in instances)

    batch_seq = np.array([inst + [pad] * (max_len - len(inst)) for inst in instances])
    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(instances, pad):
    """Pad the instance to the max seq length in batch."""
    max_len = max(len(inst) for inst in instances)

    batch_seq = np.array([inst + [pad] * (max_len - len(inst)) for inst in instances])

    return torch.tensor(batch_seq, dtype=torch.long)


def thp_collate_fn(instances, pad: int = 0):
    """
    Collate function for PyTorch dataloader.
        pad: integer to pad all sequences
    """

    time, event_type, gt_cluster, f = list(zip(*instances))
    time = pad_time(time, pad)
    event_type = pad_type(event_type, pad)

    return time, event_type, torch.tensor(gt_cluster, dtype=torch.long), torch.tensor(f, dtype=torch.long)


def download_unpack_zip(data_dict: Dict, data_dir):
    """
    Downloads zipped dataset and unpacks it to data directory
    """
    zipurl = data_dict["url"]
    res_path = "/".join(data_dir.split("/")[:-1])
    print(zipurl)
    print(res_path)
    res_code = os.system(f"wget {zipurl} -P {res_path}")
    if res_code != 0:
        print(res_code)
        raise Exception("Encountered error while downloading data")
    zip_name = zipurl.split("/")[-1]
    lfilename = os.path.join(res_path, zip_name)
    with zipfile.ZipFile(lfilename) as file:
        os.makedirs(res_path, exist_ok=True)
        file.extractall(res_path)
    if Path(os.path.join(res_path, "__MACOSX")).exists():
        shutil.rmtree(
            os.path.join(res_path, "__MACOSX"),
        )
    print(f"Successfully downloaded and unpacked data into {res_path}")
    os.remove(lfilename)
    return


def load_data(
    data_dir: Union[str, Path],
    maxlen: int = -1,
    ext: str = "txt",
    time_col: str = "time",
    event_col: str = "event",
    datetime: bool = True,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict, List[Dict], torch.Tensor, torch.Tensor]:
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    - directory containing sequences
        maxsize     - maximum number of sequences to load
        maxlen      - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         - extension of files in data_dir directory
        time_col    - title of column with timestamps
        event_col   - title of column with event types
        datetime    - variable indicating if time values in files are represented in datetime format

    Returns:
        ss          - list of torch.Tensor containing sequences. Each tensor has shape (L, 2), where
                        element is a sequence of pair (time, event type)
        Ts          - tensor of right edges T_n of intervals (0, T_n) in which point processes realizations lie.
        class2idx   - dict of event types and their indices
        user_list   - representation of sequences suitable for Cohortney
        gt_ids      - torch.Tensor of ground truth cluster labels (if available)
        freq_events - torch.Tensor of most frequent events of each sequence

    """

    sequences = []
    classes = set()
    nb_files = 0
    freq_events = []

    for file in sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    ):
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():

            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df[event_col].unique()))
            if datetime:
                df[time_col] = pd.to_datetime(df[time_col])
                df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1, "D")
            if maxlen > 0:
                df = df.iloc[:maxlen]

            sequences.append(df)
            freq_events.append(df[event_col].mode()[0])

    classes = list(classes)
    class2idx = {cls: idx for idx, cls in enumerate(classes)}
    freq_events = [class2idx[f] for f in freq_events]

    ss, Ts = [], []
    user_list = []
    for i, df in enumerate(sequences):
        user_dict = dict()
        if sequences[i][time_col].to_numpy()[-1] < 0:
            continue
        sequences[i][event_col].replace(class2idx, inplace=True)
        for event_type in class2idx.values():
            dat = sequences[i][sequences[i][event_col] == event_type]
            user_dict[event_type] = dat[time_col].to_numpy()
        user_list.append(user_dict)

        st = np.vstack(
            [sequences[i][time_col].to_numpy(), sequences[i][event_col].to_numpy()]
        )
        tens = torch.FloatTensor(st.astype(np.float32)).T

        if maxlen > 0:
            tens = tens[:maxlen]
        ss.append(tens)
        Ts.append(tens[-1, 0])

    Ts = torch.FloatTensor(Ts)
    freq_events = torch.LongTensor(freq_events)

    gt_ids = None
    if Path(data_dir, "clusters.csv").exists():
        gt_ids = pd.read_csv(Path(data_dir, "clusters.csv"))["cluster_id"].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)

    return ss, Ts, class2idx, user_list, gt_ids, freq_events


def load_data_thp(
    data_dir: Union[str, Path],
    maxlen: int = -1,
    ext: str = "csv",
    time_col: str = "time",
    event_col: str = "event",
    datetime: bool = False,
) -> Tuple[List[Dict], List, List, int, int]:
    """
    Loads the sequences saved in the given directory.
    Args:
        data_dir    - directory containing sequences
        maxsize     - maximum number of sequences to load
        maxlen      - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         - extension of files in data_dir directory
        time_col    - title of column with timestamps
        event_col   - title of column with event types
        datetime    - variable indicating if time values in files are represented in datetime format
    Returns:
        sequences          - list of torch.Tensor containing sequences. Each tensor has shape (L, 2), where
                        element is a sequence of pair (time, event type)
        gt_ids      - list of ground truth cluster labels (if available)
        freq_events - list of most frequent events of each sequence
        num_events - number of types of events in dataset
        num_clusters - number of clusters in dataset
    """

    sequences = []
    classes = set()
    freq_events = [] 

    for file in sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    ):
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():

            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df[event_col].unique()))
            if datetime:
                df[time_col] = pd.to_datetime(df[time_col])
                df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1, "D")
            else:
                df[time_col] = df[time_col] - df[time_col][0]
            if maxlen > 0:
                df = df.iloc[:maxlen]
            curr_dict = {}
            curr_dict["time_since_start"] = df[time_col].tolist()
            curr_dict["type_event"] = df[event_col].tolist()
            sequences.append(curr_dict)
            freq_events.append(df[event_col].mode()[0])

    classes = list(classes)
    if isinstance(classes[0], str):
        num_events = len(classes)
        dict_map = {}
        i = 0
        for cl in classes:
            dict_map[cl] = i
            i += 1
        for seq in sequences:
            seq["type_event"] = [dict_map[event] for event in seq["type_event"]]
        freq_events = [dict_map[f] for f in freq_events]
    else:
        num_events = int(max(classes)) + 1

    gt_ids = None
    num_clusters = -1
    if Path(data_dir, "clusters.csv").exists():
        gt_ids = pd.read_csv(Path(data_dir, "clusters.csv"))["cluster_id"].tolist()
        num_clusters = int(max(gt_ids) - min(gt_ids)) + 1

    return sequences, gt_ids, freq_events, num_events, num_clusters


def load_data_kshape(
    data_dir,
    num_events: int,
    time_col: str = "time",
    event_col: str = "event",
    ext: str = "csv",
    max_length: int = 100,
):
    """
    Loads the sequences saved in the given directory.
    Args:
        data_dir    (str, Path) - directory containing sequences
        num_events - number of different event types
        time_col - title of time column
        event_col - title of event column
        ext - extension of individual sequence file
        max_length - limit of sequence length
    """

    files_with_digits = sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    )

    # getting all event types

    all_events = set()
    seq_max_length = 0
    for file in files_with_digits:
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():
            sequence_file = pd.read_csv(Path(data_dir, file))
            seq_max_length = max(seq_max_length, len(sequence_file))
            all_events = all_events.union(set(sequence_file[event_col].unique()))

    # max len of sequence
    max_length = min(max_length, seq_max_length)
    events_arr = list(all_events)
    ts = []
    for file in files_with_digits:
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():
            sequence_file = pd.read_csv(Path(data_dir, file))
            if sequence_file[time_col].to_numpy()[-1] < 0:
                continue
            data = []
            for event_type in events_arr:
                d = np.zeros(max_length)
                dat = sequence_file[sequence_file[event_col] == event_type][
                    time_col
                ].to_numpy()
                curr_l = min(max_length, len(dat))
                d[:curr_l] = dat[:curr_l]
                data.append(d)
            ts.append(np.array(data))

    # transforming data
    ts = np.array(ts)
    ts = to_time_series_dataset(ts)
    ts_reshaped = np.zeros((len(ts), num_events, ts.shape[2]))
    for i in range(len(ts)):
        ts_reshaped[i] = ts[i][:num_events]
    print("Data processing completed")

    return ts_reshaped


def sep_hawkes_proc(user_list, event_type):
    """
    Transforming data to the array taking into account an event type
    """
    sep_seqs = []
    for user_dict in user_list:
        sep_seqs.append(np.array(user_dict[event_type], dtype=np.float32))

    return sep_seqs


def get_partition(
    sample, num_of_steps, num_of_event_types, time_col, event_col, end_time=None
):
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
    sample = sample.loc[:, [time_col, event_col]].copy()
    sample = sample.sort_values(by=[time_col])
    sample = sample.reset_index().loc[:, [time_col, event_col]].copy()
    if end_time is None:
        end_time = sample[time_col][len(sample[time_col]) - 1]

    partition = torch.zeros(num_of_steps, num_of_event_types + 1)

    # finding time stamp
    dt = end_time / num_of_steps
    partition[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    sample[time_col] = (sample[time_col] / dt).astype(int)
    mask = sample[time_col] == num_of_steps
    sample.loc[mask, time_col] -= 1

    # counting points
    sample = sample.reset_index()
    sample = sample.groupby([time_col, event_col]).count()
    sample = sample.reset_index()
    sample.columns = [time_col, event_col, "num"]
    try:
        sample[event_col] = sample[event_col].astype(int)
    except:
        global events
        global cur
        for i in range(len(sample[event_col])):
            if sample[event_col].iloc[i] not in events:
                events[sample[event_col].iloc[i]] = cur
                cur += 1
            sample[event_col].iloc[i] = events[sample[event_col].iloc[i]]
        sample[event_col] = sample[event_col].astype(int)

    # computing partition
    temp = torch.from_numpy(sample.to_numpy())
    partition[temp[:, 0], temp[:, 1] + 1] = temp[:, 2].float()
    return partition


def get_dataset(data_dir, n_classes, n_steps, time_col, event_col, ext, n_files=None):
    """
    Reads dataset
    inputs:
            data_dir - str, path to csv files with dataset
            n_classes - int, number of event types
            n_steps - int, number of steps in partitions
            time_col - title of time column
            event_col - title of event column
            ext - extension of individual sequence file
    outputs:
            data - torch.Tensor, size = (N, n_steps, n_classes + 1), dataset
            target - torch.Tensor, size = (N), true labels or None
    """
    # searching for files
    files = os.listdir(data_dir)
    target = None
    last_event_target = False

    # reading data
    files = sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    )
    # reading target
    if "clusters.csv" in files:
        files.remove("clusters.csv")
        target = torch.Tensor(
            pd.read_csv(os.path.join(data_dir, "clusters.csv"))["cluster_id"]
        )
        if n_files is not None:
            target = target[:n_files]
    if "info.json" in files:
        files.remove("info.json")
    if n_files is not None:
        files = files[:n_files]
    data = torch.zeros(len(files), n_steps, n_classes + 1)
    for i, f in tqdm.tqdm(enumerate(files)):
        df = pd.read_csv(os.path.join(data_dir, f))
        data[i, :, :] = get_partition(df, n_steps, n_classes, time_col, event_col)

    return data, target


def reshape_data_tsfresh(seq_dataset, n_classes, n_steps, settings):
    """
    Transform sequences dataset into dataset of features
    """
    len_data = seq_dataset.shape[0]
    data_divided = []
    for i in range(n_classes):
        data_divided.append(seq_dataset[:, :, i].reshape(-1))
    to_extract = []
    for i in range(n_classes):
        ids = np.arange(len_data).repeat(n_steps)
        tmp = np.vstack((ids, data_divided[i]))
        tmp = tmp.T
        to_extract.append(pd.DataFrame(data=tmp, columns=["id", "value"]))
    tfs = []
    # parameters of tsfresh features extraction
    if settings == "complete":
        settings = ComprehensiveFCParameters()
    elif settings == "efficient":
        settings = EfficientFCParameters()
    elif settings == "minimal":
        settings = MinimalFCParameters()
    for i in range(n_classes):
        tf = tsfresh.extract_features(
            to_extract[i], column_id="id", default_fc_parameters=settings
        )
        tfs.append(tf)
    data_feat = pd.concat(
        [tfs[i].reindex(tfs[0].index) for i in range(n_classes)], axis=1
    )
    print(data_feat.shape)
    data_feat.fillna(0, inplace=True)
    data_feat.replace([np.inf, -np.inf], 0, inplace=True)
    data_tensor = torch.from_numpy(data_feat.values).float()
    return data_tensor
