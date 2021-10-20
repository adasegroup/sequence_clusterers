from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
import pandas as pd
import json


class CohortneyCsvLogger(LightningLoggerBase):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.train_pd_table = pd.DataFrame(
            columns=["train_loss", "train_pur", "train_time"]
        )
        self.val_pd_table = pd.DataFrame(columns=["val_loss", "val_pur"])
        self.test_results = pd.DataFrame(columns=["test_loss", "test_pur"])

    @property
    def test_results(self):
        return self.__test_results

    @test_results.setter
    def test_results(self, res):
        self.__test_results = res

    @property
    def save_dir(self):
        return self.__save_dir

    @save_dir.setter
    def save_dir(self, save_dir):
        self.__save_dir = save_dir

    @property
    def train_pd_table(self):
        return self.__train_pd_table

    @train_pd_table.setter
    def train_pd_table(self, table):
        self.__train_pd_table = table

    @property
    def val_pd_table(self):
        return self.__val_pd_table

    @val_pd_table.setter
    def val_pd_table(self, table):
        self.__val_pd_table = table

    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self.save_dir.split("/")[-1]

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if "train_loss" in metrics:
            self.train_pd_table.loc[len(self.train_pd_table)] = [
                metrics["train_loss"],
                metrics["train_pur"],
                metrics["train_time"],
            ]
        elif "val_loss" in metrics:
            self.val_pd_table.loc[len(self.val_pd_table)] = [
                metrics["val_loss"],
                metrics["val_pur"],
            ]
        elif "test_loss" in metrics:
            self.test_results.loc[len(self.test_results)] = [
                metrics["test_loss"],
                metrics["test_pur"],
            ]

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.train_pd_table.to_csv(self.save_dir + "/training.csv")
        self.val_pd_table.to_csv(self.save_dir + "/validation.csv")
        self.test_results.to_csv(self.save_dir + "/test_results.csv")
