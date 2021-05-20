from src.dataset.random_seq import *

dataset_urls = {
    'sin_K2_C5': '',
    'Linkedin': 'https://www.dropbox.com/s/kliukm2j4mp5b94/Linkedin.zip',
    'K5_C5': 'https://www.dropbox.com/s/0r4w3umderk1ccn/K5_C5.zip',
    'K2_C5': 'https://www.dropbox.com/s/0r4w3umderk1ccn/K5_C5.zip',
}

if __name__ == '__main__':
    dataset_ex = RandomGeneratedSequences('data/sin_K2_C5', num_of_steps=1000, num_of_event_types=8)
    partitions = []
