#
# This file contains implementations of 1d-CNN model for 
# unsupervised learning of features using Deep Clustering
#
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl


class SeqCNN(pl.LightningModule):
    def __init__(self, input_size, in_channels, device):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels)
        self.fd = self.get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.fd, self.fd),
            nn.ReLU()
        )
        self.top_layer = None

    def get_feature_dim(self):
        inp = torch.zeros(3, self.in_channels, self.input_size)
        out = self.encoder(inp)
        fd = out.shape[1]
        return fd

    def forward(self, x):
        out = self.encoder(x)
        out = self.classifier(out)
        if self.top_layer is not None:
            out = self.top_layer(out)
        return out

    def training_step(self, batch, batch_idx):
        input_tensor, target = batch 
        target = target.to(self.device)
        input_tensor = input_tensor.to(self.device)

        output = self(input_tensor)
        loss = nn.CrossEntropyLoss()(output, target)
        # self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        pass
        # losses = torch.as_tensor([o['loss'] for o in outputs])
        # self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        pass


class Encoder(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size = 3, stride = 2, padding = 1),
                #nn.ReLU(),
                )

    def forward(self, x):
        out = self.encoder_conv(x)
        out = out.reshape(out.shape[0], -1)
        return out
        