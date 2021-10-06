__all__ = ["Conv1dAutoEncoder"]

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans

from src.utils import purity


class Conv1dAutoEncoder(pl.LightningModule):
    """
    Main block of convolutional event clustering
    encoder-decoder architecture allows to create meaningful representation of chortney features
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

        self.train_index = 0
        self.val_index = 0
        self.train_metric = purity
        self.val_metric = purity
        self.test_metric = purity
        self.final_labels = None

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def training_step(self, batch, batch_idx):
        x, gts = batch
        loss = torch.nn.MSELoss()(self(x), x)
        embedds = self.predict_step(x)
        preds = self.clusterize(embedds)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": gts}

    def training_epoch_end(self, outputs):
        labels = outputs[0]["preds"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            labels = torch.cat([labels, outputs[i]["preds"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        pur = self.train_metric(gt_labels, labels)
        self.log("train_pur", pur, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, gts = batch
        loss = torch.nn.MSELoss()(self(x), x)
        embedds = self.predict_step(x)
        preds = self.clusterize(embedds)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": gts}

    def validation_epoch_end(self, outputs):
        labels = outputs[0]["preds"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            labels = torch.cat([labels, outputs[i]["preds"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        pur = self.val_metric(gt_labels, labels)
        self.log("val_pur", pur, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, gts = batch
        loss = torch.nn.MSELoss()(self(x), x)
        embedds = self.predict_step(x)
        preds = self.clusterize(embedds)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": gts}

    def test_epoch_end(self, outputs):
        """
        Outputs is list of dicts returned from training_step()
        """
        labels = outputs[0]["preds"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            labels = torch.cat([labels, outputs[i]["preds"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        pur = self.test_metric(gt_labels, labels)
        self.final_labels = labels
        self.log("test_pur", pur, prog_bar=True)

    def predict_step(self, batch):
        """
        Returns embeddings
        """
        ans = self.encoder(batch)
        ans = ans.squeeze()
        ans = ans.reshape(ans.shape[0], ans.shape[1] * ans.shape[2])

        return ans

    def clusterize(self, embeddings) -> torch.LongTensor:
        """
        Performs clusterization with provided embeddings
        """
        if self.clustering == "kmeans-sklearn":
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                init="k-means++",
                max_iter=500,
                n_init=10,
                random_state=0,
            )
            # to be refactored
            # or to be used with torch-kmeans
            if embeddings.is_cuda:
                pred_y = kmeans.fit_predict(embeddings.cpu().detach().numpy())
                pred_y = torch.LongTensor(pred_y)
                pred_y = pred_y.cuda()
            else:
                pred_y = kmeans.fit_predict(embeddings)
                pred_y = torch.LongTensor(pred_y)
        else:
            raise Exception(f"Clusterization: {self.clustering} is not supported")

        return pred_y

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
