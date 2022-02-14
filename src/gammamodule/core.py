import torch
from typing import Tuple


def gamma_getter(
        size: Tuple[int, int],
        type_of_module: str = 'core'
):
    if type_of_module == 'core':
        return CoreGamma(size)
    else:
        raise Exception('Unknown gamma_module type')


class CoreGamma:
    def __init__(self, size):
        self.gamma = torch.ones(size)/size[0]
        self.size = size

    def __getitem__(self, item):
        return self.gamma[:, item]

    def __setitem__(self, item, value):
        self.gamma[:, item] = value

    def compute_gamma(self, **kwargs):
        pass

    def get_labels(self, ids=None):
        if ids is not None:
            clusters = torch.argmax(
                self.gamma[:, ids],
                dim=0,
            )
        else:
            clusters = torch.argmax(
                self.gamma,
                dim=0,
            )
        return clusters

    def reset(self, size):
        self.gamma = torch.ones(size)/size[0]
        self.size = size
