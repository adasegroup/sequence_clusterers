import torch
from torch.utils.data import Dataset, DataLoader


class CTLSTMDataset(Dataset):
    ''' Dataset class for neural hawkes data
    '''

    def __init__(self, seqs):
        self.event_seqs = []
        self.time_seqs = []

        for idx, seq in enumerate(seqs):
            self.event_seqs.append(torch.LongTensor([int(event['type_event']) for event in seq]))
            self.time_seqs.append(torch.FloatTensor([float(event['time_since_last_event']) for event in seq]))

    def __len__(self):
        return len(self.event_seqs)

    def __getitem__(self, index):
        sample = {
            'event_seq': self.event_seqs[index],
            'time_seq': self.time_seqs[index],
            'ids': index
        }

        return sample


def pad_batch_fn(batch_data):
    sorted_batch = sorted(batch_data, key=lambda x: x['event_seq'].size(), reverse=True)
    event_seqs = [seq['event_seq'].long() for seq in sorted_batch]
    time_seqs = [seq['time_seq'].float() for seq in sorted_batch]
    ids = [seq['ids'] for seq in sorted_batch]
    seqs_length = torch.LongTensor(list(map(len, event_seqs)))
    last_time_seqs = torch.stack([torch.sum(time_seq) for time_seq in time_seqs])

    event_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()

    for idx, (event_seq, time_seq, seqlen) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
        event_seqs_tensor[idx, :seqlen] = torch.LongTensor(event_seq)
        time_seqs_tensor[idx, :seqlen] = torch.FloatTensor(time_seq)

    return event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length, ids
