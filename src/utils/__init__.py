from src.utils.base import *
from src.utils.metrics import *


def list_collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y
