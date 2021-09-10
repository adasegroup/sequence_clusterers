import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
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
    max_length: int = 100
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
    print(events_arr)
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
    print(ts.shape)
    ts = to_time_series_dataset(ts)
    ts_reshaped = np.zeros((len(ts), num_events, ts.shape[2]))
    for i in range(len(ts)):
        ts_reshaped[i] = ts[i][:num_events]
    print("Data processing completed")

    return ts_reshaped


def load_data_dmhp(
    data_dir: str,
    time_col: str = "time",
    event_col: str = "event",
    ext: str = "csv",
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict, torch.Tensor]:
    """
    Loads the sequences from the given directory.
    Returns:
        ss - list of torch.Tensor containing sequences. Each tensor has shape (L, 2), where
            element is a sequence of pair (time, event type);
        Ts - tensor of right edges T_n of intervals (0, T_n) for each point process;
        class2idx  - dict of event types and their indexes;
        gt_ids - torch.Tensor of ground truth cluster labels (if available);
        ? (not there) user_list - representation of sequences suitable for Cohortney

    """

    with open(Path(data_dir, "info.json")) as info:
        info = json.load(info)
    # seq_nmb is not used anywhere
    # seq_nmb = info["seq_nmb"]
    gt_ids = None
    if Path(data_dir, "clusters.csv").exists():
        gt_ids = pd.read_csv(Path(data_dir, "clusters.csv"))["cluster_id"].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)

    classes = info["classes"]
    class2idx = {cl: idx for idx, cl in enumerate(classes)}

    ss, Ts = [], []

    for file in sorted(
        os.listdir(data_dir),
        key=lambda x: int(re.sub(fr".{ext}", "", x))
        if re.sub(fr".{ext}", "", x).isdigit()
        else 0,
    ):
        if file.endswith(f".{ext}") and re.sub(fr".{ext}", "", file).isnumeric():

            f = pd.read_csv(Path(data_dir, file))
            if f[time_col].to_numpy()[-1] < 0:
                continue

            f[event_col].replace(class2idx, inplace=True)
            # for event_type in class2idx.values():
            #    dat = f[f[event_col] == event_type]

            st = np.vstack([f[time_col].to_numpy(), f[event_col].to_numpy()])
            tens = torch.FloatTensor(st.astype(np.float32)).T
            ss.append(tens)
            Ts.append(tens[-1, 0])

    Ts = torch.FloatTensor(Ts)
    print("Data processing completed")
    return ss, Ts, class2idx, gt_ids


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
