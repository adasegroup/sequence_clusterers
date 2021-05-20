from src.utils.base import *
from src.utils.metrics import *
from src.utils.net import *

dataset_urls = {
    'sin_K2_C5': 'https://www.dropbox.com/s/hn37oyidt9joj2p/sin_K2_C5.zip',
    'Linkedin': 'https://www.dropbox.com/s/kliukm2j4mp5b94/Linkedin.zip',
    'K5_C5': 'https://www.dropbox.com/s/0r4w3umderk1ccn/K5_C5.zip',
    'K2_C5': 'https://www.dropbox.com/s/ertguufarzvwp3l/K2_C5.zip',
}


def list_collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y
