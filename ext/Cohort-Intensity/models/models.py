import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Scaled_softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = Parameter(torch.ones(1))
    def forward(self, x):
        return self.s*torch.log(1+torch.exp(x/self.s))

class LSTM_single_point_process(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,\
                 batch_first = True, dropout = 0, bidirectional = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,\
                            batch_first = batch_first, dropout = dropout,\
                            bidirectional = bidirectional)
        if bidirectional:
            self.hidden0 = Parameter(torch.randn(num_layers*2, hidden_size))
            self.cell0 = Parameter(torch.randn(num_layers*2, hidden_size))
            self.W = Parameter(torch.randn(hidden_size*2, num_classes))
        else:
            self.hidden0 = Parameter(torch.randn(num_layers, hidden_size))
            self.cell0 = Parameter(torch.randn(num_layers, hidden_size))
            self.W = Parameter(torch.randn(hidden_size, num_classes))
        self.f = Scaled_softplus()
        self.num_classes = num_classes
        self.bidir = bidirectional
        
    def forward(self, s, provide_states = False):
        bs, seq_len, _ = s.shape
        hidden0 = self.hidden0[:,None,:].repeat(1,bs,1)
        cell0 = self.cell0[:,None,:].repeat(1,bs,1)
        out, (hidden, cell) = self.lstm(s, (hidden0, cell0))
        lambdas = torch.zeros(bs,seq_len, self.num_classes)
        if self.bidir:
            h0 = hidden0[-3:, 0, :].reshape(-1)
        else:
            h0 = hidden0[-1, 0, :].reshape(-1)
        lambdas[:,0,:] = self.f(h0 @ self.W)[None,:].repeat(bs,1)
        lambdas[:,1:,:] = self.f(out[:,:-1,:] @ self.W)
        if provide_states:
            return lambdas, hidden, cell
        return lambdas
    
    def simulate(self, batch_size, dt, seq_len, verbose = False):
        with torch.no_grad():
            self.eval()
            res = torch.zeros(batch_size, seq_len, 1+self.num_classes)
            hidden0 = self.hidden0[:,None,:]
            cell0 = self.cell0[:,None,:]
            for b in range(batch_size):
                if verbose == True:
                    print('Generating batch {}/{}'.format(b+1, batch_size))
                for s in range(seq_len):
                    if verbose and s%100==0:
                        print('>>> Generating sequence step {}/{}'.format(s+1, seq_len))
                    if s == 0:
                        if self.bidir:
                            h0 = hidden0[-3:, :, :].reshape(-1)
                        else:
                            h0 = hidden0[-1, :, :].reshape(-1)
                        lambdas = self.f(h0 @ self.W)
                        res[b, s, 1:] = torch.poisson(lambdas*dt[b])
                        res[b, s, 0]  = dt[b]
                        hidden = hidden0.clone()
                        cell = cell0.clone()
                    else:
                        o, (hidden, cell) = self.lstm(res[b, s-1][None,None,:], (hidden, cell))
                        lambdas = self.f(o[0,-1,:] @ self.W)
                        res[b, s, 1:] = torch.poisson(lambdas*dt[b])
                        res[b, s, 0]  = dt[b]
            return res
        
class LSTM_cluster_point_processes(nn.Module):        
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_clusters, n_steps,\
                 batch_first = True, dropout = 0, bidirectional = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,\
                            batch_first = batch_first, dropout = dropout,\
                            bidirectional = bidirectional)
        self.bn = nn.BatchNorm1d(n_steps)
        if bidirectional:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers*2, hidden_size))
            self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers*2, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size*2, num_classes))
        else:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size, num_classes))
        for k in range(num_clusters):
            setattr(self, 'f_{}'.format(k), Scaled_softplus())
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.bidir = bidirectional
    
    def init_weigh(self, *args):
        tmp = self.hidden_size**0.5
        return torch.rand(*args)*2/tmp-1/tmp
    def forward(self, s, provide_states = False):
        bs, seq_len, _ = s.shape
        lambdas = torch.zeros(self.num_clusters, bs, seq_len, self.num_classes)
        for k in range(self.num_clusters):
            hidden0 = self.hidden0[k,:,None,:].repeat(1,bs,1)
            cell0 = self.cell0[k,:,None,:].repeat(1,bs,1)
            out, (hidden, cell) = self.lstm(s, (hidden0, cell0))
            out = self.bn(out)
            if self.bidir:
                h0 = hidden0[-3:, 0, :].reshape(-1)
            else:
                h0 = hidden0[-1, 0, :].reshape(-1)
            lambdas[k,:,0,:] = getattr(self,'f_{}'.format(k))(h0 @ self.W)[None,:].repeat(bs,1)
            lambdas[k,:,1:,:] = getattr(self,'f_{}'.format(k))(out[:,:-1,:] @ self.W)
        if provide_states:
            return lambdas, hidden, cell
        return lambdas
        
        
        
        
        
        
        
        
        