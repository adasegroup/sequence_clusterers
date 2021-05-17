import torch
import torch.nn as nn

class DoubleLinear(nn.Module):
    def __init__(self, in_size_1, in_size_2, out_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn((out_size, in_size_1)))
        self.U = nn.Parameter(torch.randn((out_size, in_size_2)))
        self.d = nn.Parameter(torch.randn(out_size))
    def forward(self, k, h):
        return self.W @ k  + self.U @ h + self.d

class Extra_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        return 2*self.sigm(x) - 1

class Scaled_Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.Tensor([1]))
    def forward(self, x):
        return self.s * torch.log(1 + torch.exp(x/self.s))

class Scaled_MultiSoftplus(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = nn.Parameter(torch.ones(s))
    def forward(self, x):
        return self.s * torch.log(1 + torch.exp(x/self.s))

class ContTimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        #i
        self.i = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_i = nn.Sigmoid()
        #i_hat
        self.i_hat = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_i_hat = nn.Sigmoid()
        #f
        self.f = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_f = nn.Sigmoid()
        #f_hat
        self.f_hat = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_f_hat = nn.Sigmoid()
        #z
        self.z = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_z = Extra_Sigmoid()
        #o
        self.o = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_o = nn.Sigmoid()
        #remebered_o
        self.remembered_o = torch.Tensor(torch.zeros(hidden_size))
        #delta
        self.delta = DoubleLinear(input_size, hidden_size, hidden_size)
        self.sigm_delta = Scaled_Softplus()
        self.prev_delta = torch.Tensor(torch.zeros(hidden_size))
        #c
        self.c = torch.Tensor(torch.zeros(hidden_size))
        #c_hat
        self.c_hat = torch.Tensor(torch.zeros(hidden_size))
        #lambdas
        self.lambdas = nn.Linear(hidden_size, input_size-1, False)
        self.sigm_lambdas = Scaled_MultiSoftplus(input_size - 1)
    def update_c(self, delta, delta_time):
        self.c = self.c_hat + (self.c - self.c_hat) * torch.exp(-delta * delta_time)
    def get_lambdas(self, seq):
        prev_time = 0
        lambdas = []
        first = True
        for event in seq:
            cur_time = event[0]
            k = event[1]
            #computing h
            self.update_c(self.prev_delta, cur_time - prev_time)
            h = self.remembered_o * (2*torch.sigmoid(self.c) - 1)
            #computing gate
            i = self.sigm_i(self.i(k, h))
            i_hat = self.sigm_i_hat(self.i_hat(k, h))
            f = self.sigm_f(self.f(k, h))
            f_hat = self.sigm_f_hat(self.f_hat(k, h))
            z = self.sigm_z(self.z(k, h))
            o = self.sigm_o(self.o(k, h))
            self.remembered_o = o
            #updating c
            self.c = f * self.c + i * z
            self.c_hat = f_hat * self.c_hat + i_hat * z
            self.prev_delta = self.sigm_delta(self.delta(k, h))
            l = self.sigm_lambdas(self.lambdas(h))
            if first:
                first = False
                continue
            lambdas.append(l[torch.argmax(k)-1])
        return lambdas
    def get_lambdas_at_t(self, seq, t):
        prev_time = 0
        for event in seq:
            cur_time = event[0]
            k = event[1]
            if cur_time > t:
                break
            #computing h
            self.update_c(self.prev_delta, cur_time - prev_time)
            h = self.remembered_o * (2*torch.sigmoid(self.c) - 1)
            #computing gate
            i = self.sigm_i(self.i(k, h))
            i_hat = self.sigm_i_hat(self.i_hat(k, h))
            f = self.sigm_f(self.f(k, h))
            f_hat = self.sigm_f_hat(self.f_hat(k, h))
            z = self.sigm_z(self.z(k, h))
            o = self.sigm_o(self.o(k, h))
            self.remembered_o = o
            #updating c
            self.c = f * self.c + i * z
            self.c_hat = f_hat * self.c_hat + i_hat * z
            self.prev_delta = self.sigm_delta(self.delta(k, h))
        self.update_c(self.prev_delta, t - prev_time)
        h = self.remembered_o * (2*torch.sigmoid(self.c) - 1)
        l = self.sigm_lambdas(self.lambdas(h))
        return l
