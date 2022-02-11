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
    load_data_thp,
    thp_collate_fn,
)


class THPDataset(Dataset):
    """Event stream dataset."""

    def __init__(
        self,
        list_of_dicts,
        gt_labels,
        freq_events,
        indices,
        num_events,
        num_clusters,
        maxsize: Optional[int] = None,
    ):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        super(THPDataset, self).__init__()
        self.time = [list_of_dicts[i]["time_since_start"] for i in indices]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [
            [elem + 1 for elem in list_of_dicts[i]["type_event"]] for i in indices
        ]
        self.target = [gt_labels[i] for i in indices]
        self.freqevent = [freq_events[i] for i in indices]
        if maxsize is not None:
            self.time = self.time[:maxsize]
            self.event_type = self.event_type[:maxsize]
            self.target = self.target[:maxsize]
            self.freqevent = self.freqevent[:maxsize]

        self.length = len(self.target)
        self.num_events = num_events
        self.num_clusters = num_clusters

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Each returned element is a list, which represents an event stream"""
        return self.time[idx], self.event_type[idx], self.target[idx], self.freqevent[idx]


class THPDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_config_yaml: Union[str, Path] = "./",
        maxsize: Optional[int] = None,
        maxlen: int = -1,
        train_val_split: float = 0.75,
        ext: str = "csv",
        time_col: str = "time",
        event_col: str = "event",
        datetime: bool = False,
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
        self.train_val_split = train_val_split
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.datetime = datetime
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.labels = None

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
        data, gt_ids, freq_events, num_events, num_clusters = load_data_thp(
            self.data_dir,
            self.maxlen,
            self.ext,
            self.time_col,
            self.event_col,
            self.datetime,
        )
        self.dataset = data
        self.labels = gt_ids
        self.freq_events = freq_events

    def setup(self, stage: Optional[str] = None):
        """
        Assign train/val datasets for use in dataloaders
        """
        if stage == "fit" or stage is None:
            permutation = np.random.permutation(len(self.dataset))
            split = int(self.train_val_split * len(self.dataset))
            train_indices = permutation[:split]
            val_indices = permutation[split:]
            self.train_data = THPDataset(
                self.dataset,
                self.labels,
                self.freq_events,
                train_indices,
                self.num_events,
                self.num_clusters,
                self.maxsize,
            )
            self.val_data = THPDataset(
                self.dataset,
                self.labels,
                self.freq_events,
                val_indices,
                self.num_events,
                self.num_clusters,
                self.maxsize,
            )

        # Assign test dataset for use in dataloader
        if stage == "test":
            test_indices = range(0, len(self.dataset))
            self.test_data = THPDataset(
                self.dataset,
                self.labels,
                self.freq_events,
                test_indices,
                self.num_events,
                self.num_clusters,
                self.maxsize,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=thp_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=thp_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=thp_collate_fn,
        )
