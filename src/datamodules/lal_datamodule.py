from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.gammamodule.core import gamma_getter
from src.utils.preprocessing import preprocess
import numpy as np
import torch


class LalDataset(Dataset):
    def __init__(self, data, target, n_clusters, gamma_module):
        self.data = data
        self.target = target
        self.gamma = gamma_getter((n_clusters, data.shape[0]), gamma_module)

    def __getitem__(self, item):
        d = self.data[item]
        t = self.target[item]
        g = self.gamma[item]
        return d, t, g

    def __len__(self):
        return len(self.data)

    def reset_gamma(self, n_clusters):
        self.gamma.reset((n_clusters, self.data.shape[0]))


class LALDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str,
            train_val_split: float = 0.9,
            max_computing_size: Optional[int] = None,
            gamma_module: str = 'core',
            preprocessing: dict = {'type': 'equipartition',
                                   'n_steps': 128,
                                   'n_classes': 5},
            n_clusters: int = 1,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.max_computing_size = max_computing_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.gamma_module = gamma_module
        self.n_clusters = n_clusters
        self.preprocessing = preprocessing
        self.data = None
        self.target = None

        # self.dims is returned when you call datamodule.size()
        # TODO dims
        self.dims = None

        self.dataset: Optional[LalDataset] = None  # TODO optimize

        self.data_train: Optional[LalDataset] = None
        self.data_val: Optional[LalDataset] = None
        self.test_data: Optional[LalDataset] = None
            
    def prepare_data(self):
        data, target = preprocess(self.data_dir, self.preprocessing)
        self.data = data
        self.target = target
        self.setup()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.test_data."""
        data, target = self.data, self.target
        self.dataset = LalDataset(data, target, self.n_clusters, self.gamma_module)
        N = self.max_computing_size if self.max_computing_size is not None else len(self.dataset)
        permutation = np.random.permutation(len(self.dataset))
        pre_data_test = self.dataset[permutation[:N]]
        self.test_data = LalDataset(pre_data_test[0], pre_data_test[1], self.n_clusters, self.gamma_module)
        self.reset_datasets()

    def reset_datasets(self):
        # TODO optimize
        N = self.max_computing_size if self.max_computing_size is not None else len(self.dataset)
        split = int(self.train_val_split * N)
        permutation = np.random.permutation(len(self.dataset))
        pre_data_train = self.dataset[permutation[:split]]
        self.data_train = LalDataset(pre_data_train[0], pre_data_train[1], self.n_clusters, self.gamma_module)
        pre_data_val = self.dataset[permutation[split:N]]
        self.data_val = LalDataset(pre_data_val[0], pre_data_val[1], self.n_clusters, self.gamma_module)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        N = self.max_computing_size if self.max_computing_size is not None else len(self.dataset)
        return DataLoader(
            dataset=self.test_data,
            batch_size=N,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
