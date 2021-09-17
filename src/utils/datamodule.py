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

from pytorch_lightning import LightningDataModule

DATASET_URLS = {
    "sin_K2_C5": "https://www.dropbox.com/s/hn37oyidt9joj2p/sin_K2_C5.zip",
    "Linkedin": "https://www.dropbox.com/s/kliukm2j4mp5b94/Linkedin.zip",
    "K5_C5": "https://www.dropbox.com/s/0r4w3umderk1ccn/K5_C5.zip",
    "K2_C5": "https://www.dropbox.com/s/ertguufarzvwp3l/K2_C5.zip",
}


class CohortneyDataModule(LightningDataModule):
    def __init__(self, data_dir: Union[str, Path] = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


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


def load_data(
    data_dir: Union[str, Path],
    maxsize: Optional[int] = None,
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

    return ss, Ts, class2idx, user_list, gt_ids


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


# def download_link(url, destination="data"):
#     res_code = os.system(f"wget {url} -P {destination}")
#     if res_code != 0:
#         raise Exception("Download data with some problem")


# def unpack(lfilename, dir):
#     with zipfile.ZipFile(lfilename) as file:
#         os.makedirs(dir, exist_ok=True)
#         file.extractall(dir)
#     if Path(os.path.join(dir, "__MACOSX")).exists():
#         shutil.rmtree(
#             os.path.join(dir, "__MACOSX"),
#         )
#     print(f"Successfully downloaded and unpacked data into {dir}")
#     os.remove(lfilename)


def download_dataset(data_dir: Union[str, Path], data_name: str):
    """
    Download dataset if it is available
    :return:
    """
    if Path(data_dir).exists():
        print("Data is already in place")
        return
    else:
        download_unpack_zip(DATASET_URLS[data_name], data_dir)
