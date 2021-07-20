"""
    This file contains 
    based models for intensity prediction
"""

# imports
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.matmul(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        #linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        lin = 0 
        return energy, lin
# modules
class ScaledSoftplus(nn.Module):
    def __init__(self):
        """
           input:
                  None
           model parameters:
                  s - softplus scaling coefficient, trainable
        """
        super().__init__()
        self.s = Parameter(torch.ones(1))

    def forward(self, x):
        """
           forward pass

           input:
                  x - torch.Tensor

           output:
                  scaled_softplus(x) - torch.Tensor, shape = x.shape
        """
        return self.s * torch.log(1 + torch.exp(x / self.s))



class TransformerMultiplePointProcesses(nn.Module):
    """
        Multiple Point Processes Model, Point Processes are distinguished with different initial hidden states
    """

    def __init__(self, attention, input_size, hidden_size, num_layers, num_classes, num_clusters, n_steps,
                 batch_first=True, dropout=0, bidirectional=False):
        """
           input:
                  input_size - int, input size of the data for LSTM
                  hidden_size - int, LSTM hidden state size
                  num_layers - int, number of LSTM layers
                  num_classes - int, number of types of events that can occur
                  num_clusters - int, number of different point processes
                  n_steps - int, sequence length (used for batch normalization)
                  batch_first - bool, whether the batch should go first in LSTM
                  dropout - float (>=0,<1), dropout probability for all LSTM layers but the last one
                  bidirectional - bool, bidirectional LSTM or not

           model parameters:
                  hidden_size - int, LSTM hidden state size
                  lstm - torch.nn.Module, LSTM model
                  bn - torch.nn.Module, Batch Normalization
                  hidden0 - torch.nn.parameter.Parameter, initial hidden states, size[0] = num_clusters
                  cell0 - torch.nn.parameter.Parameter, initial cell states, size[0] = num_clusters
                  W - torch.nn.parameter.Parameter, weighs for mapping hidden state to lambda
                  f_{k} - torch.nn.Module, Scaled Softplus, k - number of point process, 0<=k<num_clusters
                  num_classes - int, number of types of events that can occur
                  num_clusters - int, number of different point processes
                  bidir - bool, bidirectional LSTM or not
        """
        super().__init__()
        self.attention = attention
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.bn = nn.BatchNorm1d(n_steps)
        if bidirectional:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers * 2, hidden_size))
            #self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers * 2, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size * 2, num_classes))
        else:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            #self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size, num_classes))
        for k in range(num_clusters):
            setattr(self, 'f_{}'.format(k), ScaledSoftplus())
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.bidir = bidirectional

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
        assert self.num_clusters > 1

        # preparing templates for new hidden and cell states
        hidden0 = torch.Tensor(self.hidden0).to(device)
        #cell0 = torch.Tensor(self.cell0).to(device)

        # merging
        hidden0[cluster_0, :, :] = (hidden0[cluster_0, :, :] + hidden0[cluster_1, :, :]) / 2
        hidden0 = torch.index_select(hidden0, 0,
                                     torch.Tensor([k for k in range(self.num_clusters) if k != cluster_1]).long())
        #cell0[cluster_0, :, :] = (cell0[cluster_0, :, :] + cell0[cluster_1, :, :]) / 2
        #cell0 = torch.index_select(cell0, 0,
                                   #torch.Tensor([k for k in range(self.num_clusters) if k != cluster_1]).long())

        # updating states
        self.hidden0 = Parameter(hidden0)
        #self.cell0 = Parameter(cell0)

        # updating number of clusters
        self.num_clusters -= 1

        # updating activation functions
        for k in range(cluster_1, self.num_clusters):
            getattr(self, 'f_{}'.format(k)).s = Parameter(torch.Tensor(getattr(self, 'f_{}'.format(k + 1)).s))

    def delete_cluster(self, cluster, device):
        # checking that deleting is important
        assert self.num_clusters > 1

        # preparing templates
        hidden0 = torch.Tensor(self.hidden0).to(device)
        #cell0 = torch.Tensor(self.cell0).to(device)

        # deleting
        hidden0 = torch.index_select(hidden0, 0,
                                     torch.Tensor([k for k in range(self.num_clusters) if k != cluster]).long())
        #cell0 = torch.index_select(cell0, 0, torch.Tensor([k for k in range(self.num_clusters) if k != cluster]).long())

        # updating states
        self.hidden0 = Parameter(hidden0)
        #self.cell0 = Parameter(cell0)

        # updating number of clusters
        self.num_clusters -= 1

        # updating activation functions
        for k in range(cluster, self.num_clusters):
            getattr(self, 'f_{}'.format(k)).s = Parameter(torch.Tensor(getattr(self, 'f_{}'.format(k + 1)).s))

    def split_cluster(self, cluster, device):
        # preparing templates
        hidden0 = torch.zeros(self.hidden0.shape[0] + 1, self.hidden0.shape[1], self.hidden0.shape[2]).to(device)
        #cell0 = torch.Tensor(self.cell0.shape[0] + 1, self.cell0.shape[1], self.cell0.shape[2]).to(device)

        # filling in non-splitting clusters
        for k in range(self.num_clusters):
            if k != cluster:
                hidden0[k, :, :] = self.hidden0[k, :, :]
                #cell0[k, :, :] = self.cell0[k, :, :]

        # splitting
        a = np.random.beta(1, 1)
        hidden0[cluster, :, :] = 2 * a * self.hidden0[cluster, :, :]
        hidden0[-1, :, :] = 2 * (1 - a) * self.hidden0[cluster, :, :]

        #cell0[cluster, :, :] = 2 * a * self.cell0[cluster, :, :]
        #cell0[-1, :, :] = 2 * (1 - a) * self.cell0[cluster, :, :]

        # updating states
        self.hidden0 = Parameter(hidden0)
        #self.cell0 = Parameter(cell0)

        # updating number of clusters
        self.num_clusters += 1

        # updating activations
        setattr(self, 'f_{}'.format(self.num_clusters - 1), ScaledSoftplus())
        getattr(self, 'f_{}'.format(self.num_clusters - 1)).s = Parameter(
            torch.Tensor(getattr(self, 'f_{}'.format(cluster)).s))

    def forward(self, s, return_states=False):
        """
            forward pass of the model

            input:
                   s - torch.Tensor, size = (batch size, sequence length, input size)
                   return_states - bool, whether the states should be returned

            output:
                   lambdas - torch.Tensor, size = (batch size, sequence length, num_classes)
        """
        bs, seq_len, _ = s.shape

        # preparing lambdas template
        lambdas = torch.zeros(self.num_clusters, bs, seq_len, self.num_classes)

        # iterating over point processes
        hiddens = []
        #cells = []
        for k in range(self.num_clusters):
            # hidden and cell state preprocessing
            hidden0 = self.hidden0[k, :, None, :].repeat(1, bs, 1)
            #cell0 = self.cell0[k, :, None, :].repeat(1, bs, 1)

            # LSTM forward pass
            out, (hidden) = self.rnn(s, (hidden0))
            energy, linear_combination = self.attention(hidden, out, out) 
            hiddens.append(energy)
            #cells.append(cell)

            out = self.bn(out)

            # finding lambdas
            if self.bidir:
                h0 = hidden0[-3:, 0, :].reshape(-1)
            else:
                h0 = hidden0[-1, 0, :].reshape(-1)
            
            
            # first lambda doesn't depend on history and depends only on the initial hidden state
            lambdas[k, :, 0, :] = getattr(self, 'f_{}'.format(k))(h0 @ self.W)[None, :].repeat(bs, 1)
            lambdas[k, :, 1:, :] = getattr(self, 'f_{}'.format(k))(out[:, :-1, :] @ self.W)
        if return_states:
            return lambdas, hiddens
        return lambdas
