import pandas
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()


class Dataset:
    def __init__(self, config, data, subset):
        data = pandas.read_csv(f"{data}/{int(subset)}.csv")
        self.subset = subset
        self.time = list(data["time"])
        self.event = list(data["event"])
        print(self.event)
        self.event_class = len(np.unique(np.array(self.event)[:-1]))
        print(self.event)
        self.config = config
        self.seq_len = data.shape[0]
        self.time_seqs, self.event_seqs = [self.time], [self.event]
        self.statistic()

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)
        print(events)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        # for i in range(10):
        #     print(self.time_seqs[i], "\n", self.event_seqs[i])
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight
