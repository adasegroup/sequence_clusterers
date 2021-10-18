import os
import re
import zipfile
import shutil

import torch
from pathlib import Path
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
from tslearn.utils import to_time_series_dataset


def download_unpack_zip(zipurl: str, data_dir):
    """
    Downloads zipped dataset and unpacks it to data directory
    """
    res_path = "/".join(data_dir.split("/")[:-1])
    # download_link(zipurl, destination=res_path)
    res_code = os.system(f"wget {zipurl} -P {res_path}")
    if res_code != 0:
        raise Exception("Encountered error while downloading data")
    zip_name = zipurl.split("/")[-1]
    # unpack(lfilename=os.path.join(res_path, zip_name), dir=res_path)
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
        Ts          - tensor of right edges T_n of intervals (0, T_n) in which point processes realizations lie.
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

    return ss, Ts, class2idx, user_list, gt_ids


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
    transforming data to the array taking into account an event type
    :param user_list:
    :param event_type:
    :return:
    """
    sep_seqs = []
    for user_dict in user_list:
        sep_seqs.append(np.array(user_dict[event_type], dtype=np.float32))

    return sep_seqs


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
        end_time = sample["time"][len(sample["time"]) - 1]

    partition = torch.zeros(num_of_steps, num_of_event_types + 1)

    # finding time stamp
    dt = end_time / num_of_steps
    partition[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    sample["time"] = (sample["time"] / dt).astype(int)
    mask = sample["time"] == num_of_steps
    sample.loc[mask, "time"] -= 1

    # counting points
    sample = sample.reset_index()
    sample = sample.groupby(["time", "event"]).count()
    sample = sample.reset_index()
    sample.columns = ["time", "event", "num"]
    try:
        sample["event"] = sample["event"].astype(int)
    except:
        global events
        global cur
        for i in range(len(sample["event"])):
            if sample["event"].iloc[i] not in events:
                events[sample["event"].iloc[i]] = cur
                cur += 1
            sample["event"].iloc[i] = events[sample["event"].iloc[i]]
        sample["event"] = sample["event"].astype(int)

    # computing partition
    temp = torch.from_numpy(sample.to_numpy())
    partition[temp[:, 0], temp[:, 1] + 1] = temp[:, 2].float()
    return partition
