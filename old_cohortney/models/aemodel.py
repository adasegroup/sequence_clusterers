import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class RNNModel(nn.Module):
    def __init__(
        self,
        num_emb,
        vocab_size,
        input_dim,
        emb_dim,
        hidden_dim,
        encoder_dim,
        n_layers,
        dropout=0.3,
    ):
        super(RNNModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        # dimensions of input tensor to be embedded, starting from 0
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        # input dim of lstm
        self.input_dim = input_dim - num_emb + emb_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.n_layers = n_layers
        self.alpha = 0.05

        # encoder
        self.embedding1 = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.emb_dim
        )
        self.lstm1 = nn.LSTM(
            self.input_dim, self.hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_dim, self.encoder_dim)

        # decoder
        self.fc2 = nn.Linear(self.encoder_dim, self.input_dim)
        self.lstm2 = nn.LSTM(
            self.input_dim, self.input_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.cat_linear = nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, x, hidden1, hidden2):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # encode
        x_toembed = x[:, :, : self.num_emb]
        x_rest = x[:, :, self.num_emb :]
        event_embedding = self.embedding1(x_toembed.int())
        # squeezing num_emb dim (since it is = 1)
        event_embedding = event_embedding.mean(2)
        lstm_input = torch.cat((event_embedding, x_rest), dim=-1)
        lstm_enc, hidden1 = self.lstm1(lstm_input, hidden1)
        enc = self.dropout(lstm_enc)
        # collapsing on time dimension
        enc = enc[:, -1, :]
        enc = F.relu(self.fc1(enc))
        # decode
        dec = F.relu(self.fc2(enc))
        dec = self.dropout(dec)
        # upsample - LSTM one-to-many
        lstm_dec = torch.zeros(batch_size, seq_len, self.input_dim).cuda()
        lstm2_output, hidden2 = self.lstm2(dec.view(batch_size, -1, self.input_dim), hidden2)
        lstm_dec[:, 0, :] = lstm2_output.reshape(batch_size, self.input_dim)
        for i in range(1, seq_len):
            lstm2_output, hidden2 = self.lstm2(lstm_dec[:, i-1,:].clone().view(batch_size, -1, self.input_dim), hidden2)
            lstm_dec[:, i, :] = lstm2_output.reshape(batch_size, self.input_dim)

        cat_logits = self.cat_linear(lstm_dec[:, :, : self.emb_dim])
        # most likely categorical variable
        cat_pred = torch.argmax(cat_logits, axis=-1)
        cat_pred = torch.unsqueeze(cat_pred, -1)
        noncat_pred = lstm_dec[:, :, self.emb_dim :]

        return cat_pred, noncat_pred, cat_logits, hidden1, hidden2

    def encode(self, x, hidden1, hidden2):
        batch_size = x.size(0)
        # encode
        x_toembed = x[:, :, : self.num_emb]
        x_rest = x[:, :, self.num_emb :]
        event_embedding = self.embedding(x_toembed.int())
        event_embedding = event_embedding.mean(2)
        lstm_input = torch.cat((event_embedding, x_rest), dim=-1)
        lstm_enc, hidden1 = self.lstm1(lstm_input, hidden1)
        lstm_enc = lstm_enc.contiguous().view(-1, self.hidden_dim)
        enc = self.dropout(lstm_enc)
        # collapsing on time dimension
        enc = enc[:, -1, :]
        enc = F.relu(self.fc1(enc))
        return enc

    def init_hidden(self, batch_size, gpu=False):
        weight = next(self.parameters()).data

        if gpu:
            hidden1 = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            )
            hidden2 = (
                weight.new(self.n_layers, batch_size, self.input_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.input_dim).zero_().cuda(),
            )
        else:
            hidden1 = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            )
            hidden2 = (
                weight.new(self.n_layers, batch_size, self.input_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.input_dim).zero_(),
            )

        return hidden1, hidden2

    def total_loss(self, cat_logits, noncat_pred, target):
        """
        Total loss is a weighted sum of classification loss for dis-embedded
        variables and MSE loss for continuous variables
        """
        cat_target = target[:, :, : self.num_emb]
        noncat_target = target[:, :, self.num_emb :]
        # categorical (classificaiont) loss
        cat_criterion = nn.CrossEntropyLoss()
        # noncategorical (regression) loss
        noncat_criterion = nn.MSELoss()
        cat_loss = cat_criterion(
            cat_logits.view(-1, self.vocab_size), cat_target.view(-1).long()
        )
        noncat_loss = noncat_criterion(noncat_pred, noncat_target.float())
        return self.alpha * noncat_loss + cat_loss
