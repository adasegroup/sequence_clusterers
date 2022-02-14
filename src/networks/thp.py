__all__ = ["TransformerHP"]

import math
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

from src.utils.thp_utils import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    LabelSmoothingLoss,
    log_likelihood,
    get_non_pad_mask,
    get_attn_key_pad_mask,
    get_subsequent_mask,
    time_loss,
    type_loss,
)


# todo: import pad parameter from config
global constant_pad
constant_pad = 0


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, enc_dev, dropout
    ):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=enc_dev,
        )

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=constant_pad)

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    normalize_before=False,
                )
                for _ in range(n_layers)
            ]
        )

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """Encode event sequences via masked self-attention."""

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=event_type, seq_q=event_type, pad=constant_pad
        )
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
        return enc_output


class Predictor(nn.Module):
    """Prediction of next event type."""

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False
        )
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class TransformerHP(pl.LightningModule):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        d_model=256,
        d_rnn=128,
        d_inner=1024,
        n_layers=4,
        n_head=4,
        d_k=64,
        d_v=64,
        num_clusters=1,
        num_types=1,
        dropout=0.1,
        smooth=0.1,
        lr=1e-4,
        integral_biased: bool = True,
        use_additional_rnn: bool = True,
        clustering_method: str = "kmeans",
    ):
        super().__init__()
        self.clustering = clustering_method
        self.integral_biased = integral_biased
        self.use_additional_rnn = use_additional_rnn
        self.train_metric = purity
        self.val_metric = purity
        self.test_metric = purity
        self.final_labels = None
        self.smooth = smooth
        self.lr = lr
        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            enc_dev=self.device,
            dropout=dropout,
        )

        self.num_types = num_types
        self.num_clusters = num_clusters
        self.train_metric = purity
        self.val_metric = purity
        self.test_metric = purity
        self.final_labels = None
        # training start
        self.time_start = time.time()
        # time pred loss is usually large, scale it to stabilize training
        self.scale_time_loss = 100

        # prediction loss function, either cross entropy or label smoothing
        if self.smooth > 0:
            self.pred_loss_func = LabelSmoothingLoss(smooth, num_types, ignore_index=-1)
        else:
            self.pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        if self.use_additional_rnn is True:
            self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type, pad=constant_pad)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        # OPTIONAL recurrent layer, this sometimes helps:
        if self.use_additional_rnn is True:
            enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

    def predict_step(self, x):
        """
        Returns cluster lables from embeddings
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
        x_time, x_event, gts, _ = batch
        latent, prediction = self(x_event, x_time)
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(
            self, latent, x_event, x_event, constant_pad
        )
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = type_loss(
            prediction[0], x_event, self.pred_loss_func
        )

        # time prediction
        se = time_loss(prediction[1], x_time)

        loss = event_loss + pred_loss + se / self.scale_time_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "gts": gts, "latent": latent}

    def training_epoch_end(self, outputs):
        latent = outputs[0]["latent"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            latent_dim1 = latent.shape[1]
            outputs_dim1 = outputs[i]["latent"].shape[1]
            if outputs_dim1 >= latent_dim1:
                outputs[i]["latent"] = outputs[i]["latent"][:,:latent_dim1,:]
            else:
                pad1dim = (0, 0, 0, latent_dim1 - outputs_dim1, 0, 0)
                outputs[i]["latent"] = F.pad(outputs[i]["latent"], pad1dim, "constant", 0)
            latent = torch.cat([latent, outputs[i]["latent"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        labels = self.predict_step(latent)
        pur = self.train_metric(gt_labels, labels)
        self.log("train_time", time.time() - self.time_start, prog_bar=False)
        self.log("train_pur", pur, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x_time, x_event, gts, _ = batch
        latent, prediction = self(x_event, x_time)
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(
            self, latent, x_event, x_event, constant_pad
        )
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = type_loss(
            prediction[0], x_event, self.pred_loss_func
        )

        # time prediction
        se = time_loss(prediction[1], x_time)

        loss = event_loss + pred_loss + se / self.scale_time_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "gts": gts, "latent": latent}

    def validation_epoch_end(self, outputs):
        latent = outputs[0]["latent"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            latent_dim1 = latent.shape[1]
            outputs_dim1 = outputs[i]["latent"].shape[1]
            if outputs_dim1 >= latent_dim1:
                outputs[i]["latent"] = outputs[i]["latent"][:,:latent_dim1,:]
            else:
                pad1dim = (0, 0, 0, latent_dim1 - outputs_dim1, 0, 0)
                outputs[i]["latent"] = F.pad(outputs[i]["latent"], pad1dim, "constant", 0)
            latent = torch.cat([latent, outputs[i]["latent"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        labels = self.predict_step(latent)
        pur = self.val_metric(gt_labels, labels)
        self.log("val_pur", pur, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x_time, x_event, gts, f = batch
        latent, prediction = self(x_event, x_time)
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(
            self, latent, x_event, x_event, constant_pad
        )
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = type_loss(
            prediction[0], x_event, self.pred_loss_func
        )

        # time prediction
        se = time_loss(prediction[1], x_time)

        loss = event_loss + pred_loss + se / self.scale_time_loss
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
            latent_dim1 = latent.shape[1]
            outputs_dim1 = outputs[i]["latent"].shape[1]
            if outputs_dim1 >= latent_dim1:
                outputs[i]["latent"] = outputs[i]["latent"][:,:latent_dim1,:]
            else:
                pad1dim = (0, 0, 0, latent_dim1 - outputs_dim1, 0, 0)
                outputs[i]["latent"] = F.pad(outputs[i]["latent"], pad1dim, "constant", 0)
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
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, self.parameters()),
            self.lr,
            betas=(0.9, 0.999),
            eps=1e-05,
        )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        return [optimizer], [lr_scheduler]
