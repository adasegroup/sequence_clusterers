from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import make_grid
from src.utils.cohortney_utils import arr_func, events_tensor, multiclass_fws_array
from src.utils.data_utils import download_unpack_zip, load_data, load_data_kshape


class CohortneyDataset(Dataset):
    def __init__(self, data, target, maxsize: Optional[int] = None):
        super(CohortneyDataset, self).__init__()
        if maxsize is None:
            self.data = data
            self.target = target
        else:
            self.data = data[:maxsize]
            self.target = target[:maxsize]

    def __getitem__(self, idx):
        d = self.data[idx]
        t = self.target[idx]
        return d, t

    def __len__(self):
        return len(self.data)


class CohortneyDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_url_json: Union[str, Path] = "./",
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
        type_: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_url_json = data_url_json
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
        self.type_ = type_

    def prepare_data(self):
        """
        Script to download data if necessary
        """
        if Path(self.data_dir).exists():
            print("Data is already in place")
            return
        else:
            data_name = self.data_dir.split("/")[-1]
            # dictionary with urls to download sequence data
            with open(self.data_url_json, "r") as url_data:
                datasets_urls = json.load(url_data)
                download_unpack_zip(datasets_urls[data_name], self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """
        Transforming dataset for conv1d_encoder
        """
        print("Transforming data")
        ss, Ts, class2idx, user_list, gt_ids = load_data(
            self.data_dir,
            maxlen=self.maxlen,
            ext=self.ext,
            time_col=self.time_col,
            event_col=self.event_col,
            datetime=self.datetime,
            type_=self.type_,
        )
        # grid generation
        grid = make_grid(self.gamma, self.Tb, self.Th, self.N, self.n)
        T_j = grid[-1]
        Delta_T = np.linspace(0, grid[-1], 2 ** self.n)
        Delta_T = Delta_T[Delta_T < int(T_j)]
        delta_T = tuple(Delta_T)

        _, events_fws_mc = arr_func(user_list, T_j, delta_T, multiclass_fws_array)
        mc_batch = events_tensor(events_fws_mc)
        self.dataset = CohortneyDataset(mc_batch, gt_ids, self.maxsize)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            permutation = np.random.permutation(len(self.dataset))
            split = int(self.train_val_split * len(self.dataset))
            self.train_data = CohortneyDataset(
                self.dataset.data[permutation[:split]],
                self.dataset.target[permutation[:split]],
            )
            # self.val_data = self.dataset[permutation[split : len(self.dataset)]]
            self.val_data = CohortneyDataset(
                self.dataset.data[permutation[split : len(self.dataset)]],
                self.dataset.target[permutation[split : len(self.dataset)]],
            )

        # Assign test dataset for use in dataloader
        if stage == "test":
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


class TslearnDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        num_clusters: int = 2,
        num_events: int = 2,
        maxsize: Optional[int] = None,
        maxlen: int = -1,
        ext: str = "csv",
        time_col: str = "time",
        event_col: str = "event",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_events = num_events
        self.maxsize = maxsize
        self.maxlen = maxlen
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        data_name = self.data_dir.split("/")[-1]
        download_dataset(self.data_dir, data_name)

    def setup(self, stage: Optional[str] = None):
        print("Transforming data")
        # transforming
        ts_reshaped = load_data_kshape(
            self.data_dir,
            self.num_events,
            self.time_col,
            self.event_col,
            self.ext,
            self.maxlen,
        )
        if Path(self.data_dir, "clusters.csv").exists():
            gt_ids = pd.read_csv(Path(self.data_dir, "clusters.csv"))[
                "cluster_id"
            ].to_numpy()
            gt_ids = torch.LongTensor(gt_ids)
        else:
            gt_ids = [0] * len(ts_reshaped)
            gt_ids = torch.LongTensor(ts_reshaped)
        self.dataset = CohortneyDataset(ts_reshaped, gt_ids)
        if self.maxsize is not None:
            self.dataset = self.dataset[: self.maxsize]
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_data = self.dataset
            self.val_data = self.dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
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
