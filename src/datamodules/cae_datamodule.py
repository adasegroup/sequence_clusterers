from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import make_grid
from src.utils.cae_utils import arr_func, events_tensor, multiclass_fws_array
from src.utils.data_utils import download_unpack_zip, load_data


class CohortneyDataset(Dataset):
    def __init__(self, data, target, freqevent, maxsize: Optional[int] = None):
        super(CohortneyDataset, self).__init__()
        if maxsize is None:
            self.data = data
            self.target = target
            self.freqevent = freqevent
        else:
            self.data = data[:maxsize]
            self.target = target[:maxsize]
            self.freqevent = freqevent[:maxsize]

    def __getitem__(self, idx):
        d = self.data[idx]
        t = self.target[idx]
        f = self.freqevent[idx]
        return d, t, f

    def __len__(self):
        return len(self.data)


class CAEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_config_yaml: Union[str, Path] = "./",
        maxsize: Optional[int] = None,
        maxlen: int = -1,
        train_val_split: float = 0.8,
        ext: str = "csv",
        time_col: str = "time",
        event_col: str = "event",
        datetime: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
        gamma: float = 1.4,
        Tb: float = 7e-06,
        Th: int = 80,
        N: int = 2500,
        n: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(data_config_yaml, "r") as stream:
            self.data_config = yaml.safe_load(stream)
        data_name = self.data_dir.split("/")[-1]
        self.num_clusters = self.data_config[data_name]["num_clusters"]
        self.num_events = self.data_config[data_name]["num_events"]
        self.maxsize = maxsize
        self.maxlen = maxlen
        self.train_val_split = train_val_split
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.datetime = datetime
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gamma = gamma
        self.Tb = Tb
        self.Th = Th
        self.N = N
        self.n = n
        self.dataset = None

    def prepare_data(self):
        """
        Script to download data if necessary
        Data transform for conv1d
        """
        if Path(self.data_dir).exists():
            print("Data is already in place")
        else:
            data_name = self.data_dir.split("/")[-1]
            # dictionary with urls to download sequence data
            download_unpack_zip(self.data_config[data_name], self.data_dir)

        print("Transforming data")
        ss, Ts, class2idx, user_list, gt_ids, freq = load_data(
            self.data_dir,
            maxlen=self.maxlen,
            ext=self.ext,
            time_col=self.time_col,
            event_col=self.event_col,
            datetime=self.datetime,
        )
        # grid generation
        grid = make_grid(self.gamma, self.Tb, self.Th, self.N, self.n)
        T_j = grid[-1]
        Delta_T = np.linspace(0, grid[-1], 2 ** self.n)
        Delta_T = Delta_T[Delta_T < int(T_j)]
        delta_T = tuple(Delta_T)

        _, events_fws_mc = arr_func(user_list, T_j, delta_T, multiclass_fws_array)
        mc_batch = events_tensor(events_fws_mc)
        self.dataset = CohortneyDataset(mc_batch, gt_ids, freq, self.maxsize)

    def setup(self, stage: Optional[str] = None):
        """
        Assign train/val datasets for use in dataloaders
        """
        if stage == "fit" or stage is None:
            permutation = np.random.permutation(len(self.dataset))
            split = int(self.train_val_split * len(self.dataset))
            self.train_data = CohortneyDataset(
                self.dataset.data[permutation[:split]],
                self.dataset.target[permutation[:split]],
                self.dataset.freqevent[permutation[:split]],
            )
            self.val_data = CohortneyDataset(
                self.dataset.data[permutation[split : len(self.dataset)]],
                self.dataset.target[permutation[split : len(self.dataset)]],
                self.dataset.freqevent[permutation[split : len(self.dataset)]],
            )

        # Assign test dataset for use in dataloader
        if stage == "test":
            print(len(self.dataset))
            self.test_data = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
