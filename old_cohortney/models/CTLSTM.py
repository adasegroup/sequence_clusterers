import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class CTLSTM(nn.Module):
    """Continuous time LSTM network with decay function."""

    def __init__(self, hidden_size, type_size, batch_first=True):
        super(CTLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.type_size = type_size
        self.batch_first = batch_first
        self.num_layers = 1

        # Parameters
        # recurrent cells
        self.rec = nn.Linear(2 * self.hidden_size, 7 * self.hidden_size)
        # output mapping from hidden vectors to unnormalized intensity
        self.wa = nn.Linear(self.hidden_size, self.type_size)
        # embedding layer for valid events, including BOS
        self.emb = nn.Embedding(self.type_size + 1, self.hidden_size)

    def init_states(self, batch_size):
        self.h_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_bar = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
         gate_f,
         gate_z,
         gate_o,
         gate_i_bar,
         gate_f_bar,
         gate_delta) = torch.chunk(self.rec(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
                torch.exp(-delta_t * duration_t.view(-1, 1))

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t

    def forward(self, event_seqs, duration_seqs, batch_first=True):
        if batch_first:
            event_seqs = event_seqs.transpose(0, 1)
            duration_seqs = duration_seqs.transpose(0, 1)

        batch_size = event_seqs.size()[1]
        batch_length = event_seqs.size()[0]

        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []

        for t in range(batch_length):
            self.init_states(batch_size)
            c, self.c_bar, o_t, delta_t = self.recurrence(self.emb(event_seqs[t]), self.h_d, self.c_d, self.c_bar)
            self.c_d, self.h_d = self.decay(c, self.c_bar, o_t, delta_t, duration_seqs[t])
            h_list.append(self.h_d)
            c_list.append(c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)

        self.output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))
        return self.output

    def log_likelihood(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, batch_first=True):
        """Calculate log likelihood per sequence."""
        batch_size, batch_length = event_seqs.shape
        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size)
        lambda_k = F.softplus(self.wa(h)).transpose(0, 1)

        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            original_loglikelihood[idx] = torch.sum(torch.log(
                lambda_k[idx, torch.arange(seq_len).long(), event_seq[1:seq_len + 1]]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        if batch_first:
            sim_time_seqs = sim_time_seqs.transpose(0, 1)
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0, 1)
        simulated_likelihood = torch.zeros(batch_size)
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
            mc_coefficient = total_time / (seq_len)
            simulated_likelihood[idx] = mc_coefficient * torch.sum(
                torch.sum(sim_lambda_k[idx, torch.arange(seq_len).long(), :]))

        loglikelihood = torch.sum(original_loglikelihood - simulated_likelihood)
        return loglikelihood


class CTLSTMClusterwise(nn.Module):
    """Continuous time LSTM network with decay function."""

    def __init__(self, hidden_size, type_size, n_clusters, batch_first=True):
        super(CTLSTMClusterwise, self).__init__()

        self.hidden_size = hidden_size
        self.type_size = type_size
        self.batch_first = batch_first
        self.num_layers = 1
        self.n_clusters = n_clusters

        # Parameters
        # recurrent cells
        self.rec = nn.Linear(2 * self.hidden_size, 7 * self.hidden_size)
        # output mapping from hidden vectors to unnormalized intensity
        self.wa = nn.Linear(self.hidden_size, self.type_size)
        # embedding layer for valid events, including BOS
        self.emb = nn.Embedding(self.type_size, self.hidden_size)

        self.h_d_init = Parameter(self.init_weigh(n_clusters, hidden_size))
        self.c_d_init = Parameter(self.init_weigh(n_clusters, hidden_size))
        self.c_bar_init = Parameter(self.init_weigh(n_clusters, hidden_size))
        self.c_init = Parameter(self.init_weigh(n_clusters, hidden_size))
        self.output = [0] * self.n_clusters

    def init_weigh(self, *args):
        """
           Used for weight initialization, output ~ U(-1/sqrt(hidden_size),1/sqrt(hidden_size))

           input:
                  args - arguments, used for torch.rand, should be desired size of Tensor
           output:
                  weighs - torch.Tensor, size = args
        """
        tmp = self.hidden_size
        return torch.rand(*args) * 2 / tmp - 1 / tmp

    def merge_clusters(self, cluster_0, cluster_1, device):
        """
            Used for merging two clusters

            input:
                   cluster_0 - int, number of cluster to merge
                   cluster_1 - int, number of cluster to merge

            output:
                   None
        """
        # checking if merging is allowed
        assert self.n_clusters > 1

        # preparing templates for new hidden and cell states
        h_d_init = torch.Tensor(self.h_d_init).to(device)
        c_d_init = torch.Tensor(self.c_d_init).to(device)
        c_bar_init = torch.Tensor(self.c_bar_init).to(device)
        c_init = torch.Tensor(self.c_init).to(device)

        # merging
        h_d_init[cluster_0, :] = (h_d_init[cluster_0, :] + h_d_init[cluster_1, :]) / 2
        h_d_init = torch.index_select(h_d_init, 0,
                                      torch.Tensor([k for k in range(self.n_clusters) if k != cluster_1]).long())
        c_d_init[cluster_0, :] = (c_d_init[cluster_0, :] + c_d_init[cluster_1, :]) / 2
        c_d_init = torch.index_select(c_d_init, 0,
                                      torch.Tensor([k for k in range(self.n_clusters) if k != cluster_1]).long())
        c_bar_init[cluster_0, :] = (c_bar_init[cluster_0, :] + c_bar_init[cluster_1, :]) / 2
        c_bar_init = torch.index_select(c_bar_init, 0,
                                        torch.Tensor([k for k in range(self.n_clusters) if k != cluster_1]).long())
        c_init[cluster_0, :] = (c_init[cluster_0, :] + c_init[cluster_1, :]) / 2
        c_init = torch.index_select(c_init, 0,
                                    torch.Tensor([k for k in range(self.n_clusters) if k != cluster_1]).long())

        # updating states
        self.h_d_init = Parameter(h_d_init)
        self.c_d_init = Parameter(c_d_init)
        self.c_bar_init = Parameter(c_bar_init)
        self.c_init = Parameter(c_init)

        # updating number of clusters
        self.n_clusters -= 1
        self.output = [0] * self.n_clusters

    def delete_cluster(self, cluster, device):
        # checking that deleting is important
        assert self.n_clusters > 1

        # preparing templates
        h_d_init = torch.Tensor(self.h_d_init).to(device)
        c_d_init = torch.Tensor(self.c_d_init).to(device)
        c_bar_init = torch.Tensor(self.c_bar_init).to(device)
        c_init = torch.Tensor(self.c_init).to(device)

        # deleting
        h_d_init = torch.index_select(h_d_init, 0,
                                      torch.Tensor([k for k in range(self.n_clusters) if k != cluster]).long())
        c_d_init = torch.index_select(c_d_init, 0,
                                      torch.Tensor([k for k in range(self.n_clusters) if k != cluster]).long())
        c_bar_init = torch.index_select(c_bar_init, 0,
                                        torch.Tensor([k for k in range(self.n_clusters) if k != cluster]).long())
        c_init = torch.index_select(c_init, 0,
                                    torch.Tensor([k for k in range(self.n_clusters) if k != cluster]).long())

        # updating states
        self.h_d_init = Parameter(h_d_init)
        self.c_d_init = Parameter(c_d_init)
        self.c_bar_init = Parameter(c_bar_init)
        self.c_init = Parameter(c_init)

        # updating number of clusters
        self.n_clusters -= 1
        self.output = [0] * self.n_clusters

    def split_cluster(self, cluster, device):
        # preparing templates
        h_d_init = torch.zeros(self.h_d_init.shape[0] + 1, self.h_d_init.shape[1]).to(device)
        c_d_init = torch.zeros(self.h_d_init.shape[0] + 1, self.h_d_init.shape[1]).to(device)
        c_bar_init = torch.zeros(self.h_d_init.shape[0] + 1, self.h_d_init.shape[1]).to(device)
        c_init = torch.zeros(self.h_d_init.shape[0] + 1, self.h_d_init.shape[1]).to(device)

        # filling in non-splitting clusters
        for k in range(self.n_clusters):
            if k != cluster:
                h_d_init[k, :] = self.h_d_init[k, :]
                c_d_init[k, :] = self.c_d_init[k, :]
                c_bar_init[k, :] = self.c_bar_init[k, :]
                c_init[k, :] = self.c_init[k, :]

        # splitting
        a = np.random.beta(1, 1)
        h_d_init[cluster, :] = 2 * a * self.h_d_init[cluster, :]
        h_d_init[-1, :] = 2 * (1 - a) * self.h_d_init[cluster, :]

        c_d_init[cluster, :] = 2 * a * self.c_d_init[cluster, :]
        c_d_init[-1, :] = 2 * (1 - a) * self.c_d_init[cluster, :]

        c_bar_init[cluster, :] = 2 * a * self.c_bar_init[cluster, :]
        c_bar_init[-1, :] = 2 * (1 - a) * self.c_bar_init[cluster, :]

        c_init[cluster, :] = 2 * a * self.c_init[cluster, :]
        c_init[-1, :] = 2 * (1 - a) * self.c_init[cluster, :]

        # updating states
        self.h_d_init = Parameter(h_d_init)
        self.c_d_init = Parameter(c_d_init)
        self.c_bar_init = Parameter(c_bar_init)
        self.c_init = Parameter(c_init)

        # updating number of clusters
        self.n_clusters += 1
        self.output = [0] * self.n_clusters

    def init_states_k(self, batch_size, k):
        self.h_d = self.h_d_init[None, k, :].repeat(batch_size, 1)
        self.c_d = self.c_d_init[None, k, :].repeat(batch_size, 1)
        self.c_bar = self.c_bar_init[None, k, :].repeat(batch_size, 1)
        self.c = self.c_init[None, k, :].repeat(batch_size, 1)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
         gate_f,
         gate_z,
         gate_o,
         gate_i_bar,
         gate_f_bar,
         gate_delta) = torch.chunk(self.rec(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
                torch.exp(-delta_t * duration_t.view(-1, 1))

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t

    def forward_k(self, event_seqs, duration_seqs, k, batch_first=True):
        if batch_first:
            event_seqs = event_seqs.transpose(0, 1)
            duration_seqs = duration_seqs.transpose(0, 1)

        batch_size = event_seqs.size()[1]
        batch_length = event_seqs.size()[0]

        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []

        for t in range(batch_length):
            self.init_states_k(batch_size, k)
            c, self.c_bar, o_t, delta_t = self.recurrence(self.emb(event_seqs[t]), self.h_d, self.c_d, self.c_bar)
            self.c_d, self.h_d = self.decay(c, self.c_bar, o_t, delta_t, duration_seqs[t])
            h_list.append(self.h_d)
            c_list.append(c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)

        self.output[k] = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))
        return self.output[k]

    def forward(self, event_seq, duration_seqs, batch_first=True):
        for k in range(self.n_clusters):
            self.forward_k(event_seq, duration_seqs, k, batch_first)
        return self.output

    def get_lambdas_k(self, event_seqs, k):
        batch_size, batch_length = event_seqs.shape
        h, c, c_bar, o, delta = torch.chunk(self.output[k], 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size)
        lambda_k = F.softplus(self.wa(h)).transpose(0, 1)
        return lambda_k

    def log_likelihood_k(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, k,
                         batch_first=True):
        """Calculate log likelihood per sequence."""
        batch_size, batch_length = event_seqs.shape
        h, c, c_bar, o, delta = torch.chunk(self.output[k], 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size)
        lambda_k = F.softplus(self.wa(h)).transpose(0, 1)

        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            original_loglikelihood[idx] = torch.sum(torch.log(
                lambda_k[idx, torch.arange(seq_len).long(), event_seq[:seq_len]]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        if batch_first:
            sim_time_seqs = sim_time_seqs.transpose(0, 1)
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0, 1)
        simulated_likelihood = torch.zeros(batch_size)
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
            mc_coefficient = total_time / (seq_len)
            simulated_likelihood[idx] = mc_coefficient * torch.sum(
                torch.sum(sim_lambda_k[idx, torch.arange(seq_len - 1).long(), :]))

        loglikelihood = original_loglikelihood - simulated_likelihood
        return loglikelihood

    def log_likelihood(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, batch_first=True):
        log_likelihood = []
        for k in range(self.n_clusters):
            log_likelihood.append(
                self.log_likelihood_k(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, k,
                                      batch_first))
        return log_likelihood

    def get_lambdas(self, event_seq, duration_seq, time_cum_seqs_tensor, times_to_compute, device):
        lambdas = torch.zeros(self.n_clusters, len(times_to_compute))
        for i, t in enumerate(times_to_compute):
            event_mod = event_seq[time_cum_seqs_tensor < t].cpu()
            duration_mod = duration_seq[time_cum_seqs_tensor < t].cpu()
            event_mod = torch.cat((event_mod, torch.Tensor([0]))).long()
            try:
                duration_mod = torch.cat(
                    (duration_mod, torch.Tensor([t - time_cum_seqs_tensor[time_cum_seqs_tensor < t].cpu()[-1]])))
            except:
                duration_mod = torch.cat((duration_mod, torch.Tensor([t])))
            event_mod = event_mod[None, :].to(device)
            duration_mod = duration_mod[None, :].to(device)
            self.forward(event_mod, duration_mod, True)
            for k in range(self.n_clusters):
                ls = self.get_lambdas_k(event_mod, k).cpu()
                assert ls.shape[0] == 1
                lambdas[k, i] = ls[0, -1, 0]
        return lambdas

