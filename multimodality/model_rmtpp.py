# Implementation of RMTPP is taken from https://github.com/woshiyyya/ERPP-RMTPP
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from optimization import BertAdam


class Net(nn.Module):
    def __init__(self, config, lossweight):
        super(Net, self).__init__()
        self.config = config
        self.n_class = config.event_class
        print("Embed", self.n_class)
        self.embedding = nn.Embedding(
            num_embeddings=config.event_class + 1, embedding_dim=config.emb_dim
        )
        print("Create embed with", config.event_class, config.emb_dim, self.embedding)
        self.emb_drop = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.emb_dim + 1,
            hidden_size=config.hid_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim)
        self.mlp_drop = nn.Dropout(p=config.dropout)
        self.event_linear = nn.Linear(
            in_features=config.mlp_dim, out_features=config.event_class
        )
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1)
        self.set_criterion(lossweight)

    def set_optimizer(self, total_step, use_bert=True):
        if use_bert:
            self.optimizer = BertAdam(
                params=self.parameters(),
                lr=self.config.lr,
                warmup=0.1,
                t_total=total_step,
            )
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)

    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        self.intensity_w = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device="cpu")
        )
        self.intensity_b = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device="cpu")
        )
        self.time_criterion = self.RMTPPLoss

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(
            pred
            + self.intensity_w * gold
            + self.intensity_b
            + (
                torch.exp(pred + self.intensity_b)
                - torch.exp(pred + self.intensity_w * gold + self.intensity_b)
            )
            / self.intensity_w
        )
        return -1 * loss

    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        self.lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        print(event_embedding.shape, input_time.unsqueeze(-1).shape)
        self.hidden_state, (self.hn, cn) = self.lstm(self.lstm_input)
        mlp_output = torch.tanh(self.mlp(self.hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cpu().contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch(
            [time_tensor[:, :-1], time_tensor[:, -1]]
        )
        event_input, event_target = self.dispatch(
            [event_tensor[:, :-1], event_tensor[:, 1]]
        )

        time_logits, event_logits = self.forward(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        print("Shape of event logits", event_logits.shape)
        loss2 = self.event_criterion(
            event_logits.view(-1, event_logits.shape[1]), event_target.view(-1)
        )
        loss = self.config.alpha * loss1 + loss2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch(
            [time_tensor[:, :-1], time_tensor[:, -1]]
        )
        event_input, event_target = self.dispatch(
            [event_tensor[:, :-1], event_tensor[:, -1]]
        )
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred
