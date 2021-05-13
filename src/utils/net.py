import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch_optimizer
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR


class ScaledSoftplus(nn.Module):
    def __init__(self):
        """
           :arg:
                  None
           :param:
                  s - softplus scaling coefficient, trainable
        """
        super().__init__()
        self.s = nn.Parameter(torch.ones(1), requires_grad=False)  # check requires_grad

    def forward(self, x: torch.Tensor):
        """
           :arg:
                  x - torch.Tensor
           :param:
                  scaled_softplus(x) - torch.Tensor, shape = x.shape
        """
        return self.s * torch.log(1 + torch.exp(x / self.s))


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = torch_optimizer.RAdam(parameters, lr=hparams.lr, eps=eps,
                                          weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = torch_optimizer.Ranger(parameters, lr=hparams.lr, eps=eps,
                                           weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        rule = lambda epoch: (1 - epoch / hparams.num_epochs) ** hparams.poly_exp
        scheduler = LambdaLR(optimizer, rule)
    else:
        raise ValueError('scheduler not recognized!')

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
