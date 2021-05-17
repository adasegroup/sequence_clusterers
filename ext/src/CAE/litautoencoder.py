import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self
                 , in_channels: int
                 , n_latent_features: int):
        super().__init__()
        self.out = n_latent_features
        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size = 3),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size = 3),
                nn.BatchNorm1d(256),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size = 3),
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size = 3),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size = 3),
                nn.BatchNorm1d(32),
                nn.Conv1d(in_channels=32, out_channels=self.out, kernel_size = 3)
   
                )  
        self.decoder =  nn.Sequential(
               nn.ConvTranspose1d(in_channels=self.out, out_channels=32, kernel_size = 3),
                nn.BatchNorm1d(32), 
                nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size = 3),
                nn.BatchNorm1d(64), 
                nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size = 3),
                nn.BatchNorm1d(128), 
                nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size = 3),
                nn.BatchNorm1d(256), 
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size = 3),
                nn.BatchNorm1d(512),
                nn.ConvTranspose1d(in_channels=512, out_channels=in_channels, kernel_size = 3),
                )
        
        self.train_index = 0
        self.val_index = 0


    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)

        loss =  torch.nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def training_epoch_end(self, outputs):
        losses = torch.as_tensor([o['loss'] for o in outputs])
        self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)

        loss = torch.nn.MSELoss()(x_hat, x)
        output = {
            'val_loss': loss
            
        }
        self.log_dict(output, prog_bar=True)

        return output

    def validation_epoch_end(self, outputs):
        losses = []
        for o in outputs:
            losses.append(o['val_loss'])

        self.log_dict({
            'avg_val_loss': torch.as_tensor(losses).mean()
            
        }, prog_bar=True)

    def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.003)
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
#             'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01),
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr_K3_C5),
            'name': 'lr schedule'
        }
        return [optimizer], [lr_scheduler]
