import logging
import warnings
import random

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO):
    """Initializes python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def make_grid(gamma, T_b, T_h, N, n):
    grid = []
    for i in range(N):
        a = gamma ** i * T_b
        if (a <= T_h):
            grid.append(a)

        else:
            break
    grid = np.array(grid)

    return grid


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

log = get_logger()