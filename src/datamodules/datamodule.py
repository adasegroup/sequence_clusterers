from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.utils import make_grid
from src.utils.cohortney_utils import arr_func, events_tensor, multiclass_fws_array
from src.utils.data_utils import download_dataset, load_data, load_data_kshape


class CohortneyDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        maxsize: Optional[int] = None,
        maxlen: int = -1,
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
        self.maxsize = maxsize
        self.maxlen = maxlen
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
        self.gt_ids = None
        # self.dims =

    def prepare_data(self):
        # download
        data_name = self.data_dir.split("/")[-1]
        download_dataset(self.data_dir, data_name)

    def setup(self, stage: Optional[str] = None):
        print("Transforming data")
        # transforming
        ss, Ts, class2idx, user_list, gt_ids = load_data(
            self.data_dir,
            maxsize=self.maxsize,
            maxlen=self.maxlen,
            ext=self.ext,
            time_col=self.time_col,
            event_col=self.event_col,
            datetime=self.datetime,
            type_=self.type_,
        )
        self.gt_ids = gt_ids
        # grid generation
        grid = make_grid(self.gamma, self.Tb, self.Th, self.N, self.n)
        T_j = grid[-1]
        Delta_T = np.linspace(0, grid[-1], 2 ** self.n)
        Delta_T = Delta_T[Delta_T < int(T_j)]
        delta_T = tuple(Delta_T)

        _, events_fws_mc = arr_func(user_list, T_j, delta_T, multiclass_fws_array)
        mc_batch = events_tensor(events_fws_mc)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_data = mc_batch
            self.val_data = mc_batch

            # len_train = int(0.8*len(mc_batch))
            # len_val = len(mc_batch) - len_train
            # self.train_data, self.val_data = random_split(mc_batch, [len_train, len_val])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = mc_batch

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
        self.gt_ids = None
        # self.dims =

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
            self.gt_ids = pd.read_csv(Path(self.data_dir, "clusters.csv"))[
                "cluster_id"
            ].to_numpy()
            self.gt_ids = torch.LongTensor(self.gt_ids)
        if self.maxsize is not None:
            self.gt_ids = self.gt_ids[: self.maxsize]
            ts_reshaped = ts_reshaped[: self.maxsize]
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_data = ts_reshaped
            self.val_data = ts_reshaped

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = ts_reshaped

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
