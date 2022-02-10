#Implementation of RMTPP is taken from https://github.com/woshiyyya/ERPP-RMTPP

from util import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model_rmtpp import Net
from tqdm import tqdm
import numpy as np
import json
import os

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--files", type=str, default=["Amazon_short_seq"])
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--save_path", type=str, default="Amazon")
    #parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--emb_dim", type=int, default=4)
    parser.add_argument("--hid_dim", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--event_class", type=int, default=8)
    parser.add_argument("--verbose_step", type=int, default=350)
    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--lr", type=int, default=1e-2)
    parser.add_argument("--epochs", type=int, default=40)
    
    config = parser.parse_args()
    files = config.files
    for d in files:
        data = {}
        seq_num = len(os.listdir(d))
        quant = 0
        for t in range(seq_num):
            print("Processing", t)
            train_set = Dataset(config, data = d, subset= f'{t}')
            config.seq_len = train_set.seq_len
            if train_set.event_class > 0:
                quant += 1
                print("Starting to work on", t)
                print(config.event_class, config.seq_len)
                config.batch_size = config.seq_len
                train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=Dataset.to_features)
                weight = np.ones(config.event_class)
                if config.importance_weight:
                    weight = train_set.importance_weight()
                    print("importance weight: ", weight)
                model = Net(config, lossweight=weight)
                model.set_optimizer(total_step=len(train_loader) * config.epochs, use_bert=True)
                model.cpu()

                for epc in range(config.epochs):
                    model.train()
                    range_loss1 = range_loss2 = range_loss = 0
                    for i, batch in enumerate(tqdm(train_loader)):
                        l1, l2, l = model.train_batch(batch)

                data[t] = str(list(model.hidden_state[-1,-1,:].detach().numpy()))
        with open(f"{config.save_path}.json", "w") as outfile:
            json.dump(data, outfile)
        print("Receive embeddings for", quant+1, "seq")