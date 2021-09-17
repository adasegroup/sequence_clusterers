from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from utils.preprocessing import tune_basis_fn


class PointProcess:
    """
    Class for point process
    """

    def __init__(self, seqs: List, Tns: torch.Tensor, eps: float, tune: bool):
        """
        seqs - list of realizations, where each item is a tensor of size (L, 2)
        Tns - tensor of Tn for each realization (each event's timestamp lies in (0, Tn))
        eps - parameter for tuning of basis functions
        tune - whether to tune or not
        """
        self.N = len(seqs)
        self.seqs = seqs
        self.Tns = Tns
        self.basis_fs = tune_basis_fn(seqs, eps, tune)

    def __iter__(self):
        for n in range(len(self.seqs)):
            cs, ts = self.seqs[n][:, 1], self.seqs[n][:, 0]
            Tn = self.Tns[n]
            assert Tn >= ts[-1]
            yield cs.long(), ts.float(), Tn
