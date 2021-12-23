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
    encoder-decoder architecture allows to create representation of Cohortney features
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
            # curr_device = torch.device(
            #    "cuda:" + str(ans.get_device()) if torch.cuda.is_available() else "cpu"
            # )
            # cluster_ids_x, cluster_centers = kmeans(
            #    X=ans,
            #    num_clusters=self.num_clusters,
            #    distance="euclidean",
            #    device=curr_device,
            # )
            kmeans = KMeans(
                n_clusters=self.num_clusters, max_iter=100, mode="euclidean", verbose=1
            )
            cluster_ids = kmeans.fit_predict(ans)
        else:
            raise Exception(f"Clusterization: {self.clustering} is not supported")
        return cluster_ids

    def training_step(self, batch, batch_idx):
        x, gts = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        preds = self.predict_step(latent)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": gts}

    def training_epoch_end(self, outputs):
        labels = outputs[0]["preds"]
        gt_labels = outputs[0]["gts"]
        for i in range(1, len(outputs)):
            labels = torch.cat([labels, outputs[i]["preds"]], dim=0)
            gt_labels = torch.cat([gt_labels, outputs[i]["gts"]], dim=0)
        pur = self.train_metric(gt_labels, labels)
        self.log("train_time", time.time() - self.time_start, prog_bar=False)
        self.log("train_pur", pur, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, gts = batch

        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        preds = self.predict_step(latent)
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
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        preds = self.predict_step(latent)
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


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformer.Layers import EncoderLayer


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


def get_non_pad_mask(seq):
    """Get the non-padding positions."""

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """For masking out the padding part of key sequence."""

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """For masking out the subsequent info, i.e., masked self-attention."""

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


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
        self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout
    ):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device("cuda"),
        )

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

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
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
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


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        num_types,
        d_model=256,
        d_rnn=128,
        d_inner=1024,
        n_layers=4,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
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

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)
