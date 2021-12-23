from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils.data_utils import (
    download_unpack_zip,
    get_dataset,
    load_data_kshape,
    reshape_data_tsfresh,
)


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


class TslearnDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_config_yaml: Union[str, Path] = "./",
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
        with open(data_config_yaml, "r") as stream:
            self.data_config = yaml.safe_load(stream)
        data_name = self.data_dir.split("/")[-1]
        self.num_clusters = self.data_config[data_name]["num_clusters"]
        self.num_events = self.data_config[data_name]["num_events"]
        self.maxsize = maxsize
        self.maxlen = maxlen
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            download_unpack_zip(self.data_config[data_name], self.data_dir)

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
        self.dataset = CohortneyDataset(ts_reshaped, gt_ids, self.maxsize)
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


class TsfreshDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_config_yaml: Union[str, Path] = "./",
        maxsize: Optional[int] = None,
        maxlen: int = -1,
        ext: str = "csv",
        time_col: str = "time",
        event_col: str = "event",
        feature_settings: str = "minimal",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(data_config_yaml, "r") as stream:
            self.data_config = yaml.safe_load(stream)
        data_name = self.data_dir.split("/")[-1]
        self.num_clusters = self.data_config[data_name]["num_clusters"]
        self.num_events = self.data_config[data_name]["num_events"]
        self.num_steps = self.data_config[data_name]["num_steps"]
        self.maxsize = maxsize
        self.maxlen = maxlen
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.feature_settings = feature_settings
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            download_unpack_zip(self.data_config[data_name], self.data_dir)

    def setup(self, stage: Optional[str] = None):
        print("Transforming data")
        # transforming
        ts_partitioned, gt_ids = get_dataset(
            self.data_dir,
            self.num_events,
            self.num_steps,
            self.time_col,
            self.event_col,
            self.ext,
        )
        gt_ids = torch.FloatTensor(gt_ids)
        ts_reshaped = reshape_data_tsfresh(
            ts_partitioned, self.num_events, self.num_steps, self.feature_settings
        )
        self.dataset = CohortneyDataset(ts_reshaped, gt_ids, self.maxsize)
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
