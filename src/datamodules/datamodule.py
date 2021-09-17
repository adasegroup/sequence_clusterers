from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.utils import make_grid
from src.utils.cohortney_utils import arr_func, events_tensor, multiclass_fws_array
from src.utils.data_utils import download_dataset, load_data


class CohortneyDataModule(LightningDataModule):
    def __init__(self, args, data_dir: Union[str, Path] = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.args = args
        self.gt_ids = None
        # self.dims =

    def prepare_data(self):
        # download
        data_name = self.data_dir.split("/")[-1]
        download_dataset(self.data_dir, data_name)

    def setup(self, stage: Optional[str] = None):
        # transforming
        ss, Ts, class2idx, user_list, gt_ids = load_data(
            self.data_dir,
            maxsize=self.args.maxsize,
            maxlen=self.args.maxlen,
            ext=self.args.ext,
            time_col="time",
            event_col="event",
            datetime=self.args.datetime,
            type_=self.args.type,
        )
        self.gt_ids = gt_ids
        # grid generation
        grid = make_grid(
            self.args.gamma, self.args.Tb, self.args.Th, self.args.N, self.args.n
        )
        T_j = grid[-1]
        Delta_T = np.linspace(0, grid[-1], 2 ** self.args.n)
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
        return DataLoader(self.train_data, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size)
