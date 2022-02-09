__all__ = ["Conv1dAutoEncoder"]

import numpy as np
import pytorch_lightning as pl
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fast_pytorch_kmeans import KMeans
from src.utils import purity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def init_weights(m):
    """
    Simple weight initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.01)


class Conv1dAutoEncoder(pl.LightningModule):
    """
    Main block of convolutional event clustering
    encoder-decoder architecture obtains representations of event sequences features
    """

    def __init__(
        self,
        in_channels: int,
        n_latent_features: int,
        num_clusters: int,
        clustering_method: str,
    ):
        super().__init__()
        self.out = n_latent_features
        self.clustering = clustering_method
        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=3),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=self.out, kernel_size=3),
        )
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.out, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(
                in_channels=512, out_channels=in_channels, kernel_size=3
            ),
        )
        self.decoder.apply(init_weights)

        self.train_index = 0
        self.val_index = 0
        self.train_metric = purity
        self.val_metric = purity
        self.test_metric = purity
        self.final_labels = None
        self.time_start = time.time()

    def forward(self, x):
        """
        Returns embeddings
        """
        latent = self.encoder(x)
        return latent

    def predict_step(self, x):
        """
        Returns cluster lables
        """
        ans = x.squeeze()
        ans = ans.reshape(ans.shape[0], ans.shape[1] * ans.shape[2])
        if self.clustering == "kmeans":
            kmeans = KMeans(
                n_clusters=self.num_clusters, max_iter=100, mode="euclidean", verbose=0
            )
            cluster_ids = kmeans.fit_predict(ans)
        else:
            raise Exception(f"Clusterization: {self.clustering} is not supported")
        return cluster_ids

    def training_step(self, batch, batch_idx):
        x, gts, _ = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "gts": gts, "latent": latent}

    def training_epoch_end(self, outputs):
        latent = outputs[0]["latent"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            latent = torch.cat([latent, outputs[i]["latent"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)

        labels = self.predict_step(latent)
        pur = self.train_metric(gt_labels, labels)
        self.log("train_time", time.time() - self.time_start, prog_bar=False)
        self.log("train_pur", pur, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, gts, _ = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "gts": gts, "latent": latent}

    def validation_epoch_end(self, outputs):
        latent = outputs[0]["latent"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            latent = torch.cat([latent, outputs[i]["latent"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)

        labels = self.predict_step(latent)
        pur = self.val_metric(gt_labels, labels)
        self.log("val_pur", pur, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, gts, f = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "gts": gts, "latent": latent, "dwn_target": f}

    def test_epoch_end(self, outputs):
        """
        Outputs is list of dicts returned from training_step()
        """
        latent = outputs[0]["latent"]
        gt_labels = outputs[0]["gts"]
        dwn_target = outputs[0]["dwn_target"]
        for i in range(1, len(outputs)):
            latent = torch.cat([latent, outputs[i]["latent"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
            dwn_target = torch.cat([dwn_target, outputs[i]["dwn_target"]], dim=0)

        labels = self.predict_step(latent)
        pur = self.test_metric(gt_labels, labels)
        self.final_labels = labels
        self.log("test_pur", pur, prog_bar=True)
        # predicting most frequent
        ans = latent.squeeze()
        features = ans.reshape(ans.shape[0], ans.shape[1] * ans.shape[2])
        N = len(features)
        split = int(0.8 * N)
        permutation = np.random.permutation(len(features))
        clf = LogisticRegression().fit(
            features.cpu().numpy()[permutation[:split]],
            dwn_target.cpu().numpy()[permutation[:split]],
        )
        self.final_probs = clf.predict_proba(features.cpu().numpy())
        self.freq_events = dwn_target

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        def adjust_lr_K3_C5(epoch):
            if epoch < 100:
                return 0.003
            if epoch >= 100 and epoch < 120:
                return 0.0003
            if epoch >= 120 and epoch < 150:
                return 0.000003
            if epoch >= 150 and epoch < 200:
                return 0.0000003
            else:
                return 0.00000003

        def adjust_lr_K5_C5(epoch):
            if epoch < 60:
                return 0.003
            if epoch >= 60 and epoch < 350:
                return 0.0003
            else:
                return 0.00003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr_K3_C5
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]
