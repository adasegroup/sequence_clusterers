__all__ = ["TsLearnClusterizer"]

from tslearn.clustering import TimeSeriesKMeans, KShape
import pytorch_lightning as pl
from src.utils.metrics import purity
import torch
import logging


logger = logging.getLogger("tslearn")


class TsLearnClusterizer(pl.LightningModule):
    """
    Main block of convolutional event clustering
    encoder-decoder architecture allows to create representation of Cohortney features
    """

    def __init__(
        self,
        num_clusters: int,
        clustering_method: str,
        kmeans_metric: str,
        max_iter: int,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_metric = kmeans_metric
        self.clustering_method = clustering_method
        self.max_iter = max_iter
        self.test_metric = purity
        self.final_labels = None
        if self.clustering_method == "k_shape":
            self.cluster_model = KShape(
                n_clusters=self.num_clusters, max_iter=self.max_iter
            )
        elif self.clustering_method == "k_means_softdtw":
            self.cluster_model = TimeSeriesKMeans(
                n_clusters=self.num_clusters,
                metric=self.kmeans_metric,
                max_iter=self.max_iter,
            )

    def forward(self, x):
        """
        Returns embeddings
        """
        pass

    def predict_step(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x, gts = batch
        preds = self.cluster_model.fit_predict(x)
        return {"preds": torch.tensor(preds), "gts": gts}

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
        logger.info(f"test purity: {pur}")
        self.final_labels = labels

    def configure_optimizers(self):
        pass
