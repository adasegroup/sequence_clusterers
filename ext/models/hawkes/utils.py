import torch

def nll_loss(model, seq):
    result = -sum([torch.log(i) for i in model.get_lambdas(seq)])
    T = seq[-1][0]
    N = int(10*T)
    tmp = 0
    for i in range(N):
        t = torch.rand(1)*T
        l = model.get_lambdas_at_t(seq,t)
        tmp += torch.sum(l)
    tmp = tmp*T/N
    result += tmp
    return result