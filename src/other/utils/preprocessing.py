import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from tslearn.utils import to_time_series_dataset

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import torch


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


def load_data_dmhp(
    data_dir: Union[str, Path],
    maxsize: Optional[int],
    maxlen: int = -1,
    ext: str = "txt",
    time_col: str = "time",
    event_col: str = "event",
    datetime: bool = True,
    type_=None,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict, List[Dict], torch.Tensor]:
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
        Ts          - tensor of right edges T_n of interavls (0, T_n) in which point processes realizations lie.
        class2idx   - dict of event types and their indices
        user_list   - representation of sequences suitable for Cohortney
        gt_ids      - torch.Tensor of ground truth cluster labels (if available)

    """

    sequences = []
    classes = set()
    nb_files = 0
    # if type_ == "booking1":
    #     time_col = "checkin"
    #     event_col = "city_id"
    # elif type_ == "booking2":
    #     time_col = "checkout"
    #     event_col = "city_id"

    for file in sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    ):
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():
            if maxsize is None or nb_files <= maxsize:
                nb_files += 1
            else:
                break

            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df[event_col].unique()))
            if datetime:
                df[time_col] = pd.to_datetime(df[time_col])
                df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1, "D")
            if maxlen > 0:
                df = df.iloc[:maxlen]

            sequences.append(df)

    classes = list(classes)
    class2idx = {cls: idx for idx, cls in enumerate(classes)}

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

    gt_ids = None
    if Path(data_dir, "clusters.csv").exists():
        gt_ids = pd.read_csv(Path(data_dir, "clusters.csv"))["cluster_id"].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)
    print("Data processing completed")

    return ss, Ts, class2idx, user_list, gt_ids


def tune_basis_fn(ss, eps=1e5, tune=True):
    """
    Tune basis functions
    """
    if tune:

        w0_arr = np.linspace(0, 15, 1000)  # 15
        all_seq = torch.cat(tuple(ss))
        T = torch.max(all_seq[:, 0]).item()
        N_events = np.sum([len(seq) for seq in ss])

        h = ((4 * torch.std(all_seq) ** 5) / (3 * N_events)) ** 0.2
        const = N_events * np.sqrt(2 * np.pi * h ** 2)
        upper_bound = lambda w: const * np.exp(-(w ** 2 * h ** 2) / 2)
        result = lambda w0: integrate.quad(upper_bound, w0, np.inf)

        basis_fs = []
        for w0 in w0_arr:
            if result(w0)[0] <= eps:
                M = int(np.ceil(T * w0 / np.pi))
                sigma = 1 / w0
                basis_fs = [
                    lambda t, m_=m: np.exp(
                        -((t - (m_ - 1) * T / M) ** 2) / (2 * sigma ** 2)
                    )
                    for m in range(1, M + 1)
                ]
                break

        print("eps =", eps, "w0 =", w0)
        print("D =", len(basis_fs))

    else:
        D = 6
        basis_fs = [lambda t: torch.exp(-(t ** 2) / (2 * k ** 2)) for k in range(D)]

    return basis_fs
