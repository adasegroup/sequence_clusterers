import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.aemodel import RNNModel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def split_seq_by_user(data, input_dim: int, len_individual: int, coldict: Dict):
    """
    Split csv dataset by id_column and transforms it into TorchTensor
        data: pandas dataframe,
        input_dim: dimension of input, incl. user_id,
        len_individual: max len of individual history
    """
    unique_ids = data[coldict["id_cols"]].unique()
    out_tensor = torch.zeros(len_individual, input_dim, 1)
    for idd in unique_ids:
        tmp_df = data[data[coldict["id_cols"]] == idd]
        # filter for booking data
        # tmp_df = tmp_df[tmp_df['diff_checkin'] < 9999]
        tmp_df = tmp_df.drop(coldict["id_cols"], inplace=False, axis=1)
        tmp_tensor = torch.from_numpy(tmp_df.values)
        tmp_tensor = tmp_tensor.unsqueeze_(-1)

        if tmp_tensor.shape[0] > len_individual:
            tmp_tensor = tmp_tensor[:len_individual]
        elif tmp_tensor.shape[0] < len_individual:
            # pad with zeros
            target = torch.zeros(len_individual, input_dim, 1)
            target[: tmp_tensor.shape[0], :, :] = tmp_tensor
            tmp_tensor = target

        out_tensor = torch.cat((out_tensor, tmp_tensor), 2)

    # tensor -> (N of id-s, len_sequence, input_dim)
    out_tensor = out_tensor.permute(2, 0, 1)
    return out_tensor


def read_data(
    filename: str, history_dim: int, input_dim: int, coldict: Dict, load: bool = False
):
    """
    Takes a csv datafile, transforms to pandas dataframe,
    selects necessary features and converts to torch tensor;
    or loads directly from pickled tensor (if load)
    """
    if load:
        seq = torch.load(filename)
        data = TensorDataset(seq.float(), seq.float())
        return data

    csv = pd.read_csv(filename)
    # recode atm data - to keep it different from zero
    csv[coldict["event_cols"]] += 1
    print(csv["event"].min(), csv["event"].max())
    csv = csv.sort_values(by=coldict["sort_cols"])
    csv = csv[coldict["necess_cols"]]
    assert csv.shape[1] - 1 == input_dim, "not correct input dimension"

    # for col in coldict['scale_cols']:
    #    csv[col] = MinMaxScaler().fit_transform(csv[col])
    csv[coldict["scale_cols"]] = MinMaxScaler().fit_transform(
        csv[coldict["scale_cols"]]
    )
    seq = split_seq_by_user(csv, input_dim, history_dim, coldict)
    data = TensorDataset(seq.float(), seq.float())
    tensorname = filename.split(".")[0]
    torch.save(seq.float(), tensorname + "_seqlen" + str(history_dim) + ".pt")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history_dim", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--num_emb", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=100000)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--encoder_dim", type=int, default=10)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--clip", type=int, default=5)  # gradient clipping
    parser.add_argument("--learning_rate", type=float, default=0.0008)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/atm_train_day_seqlen300.pt",
        # default="data/booking_challenge_tpp_labeled.csv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/atm_test_day_seqlen300.pt",
        # default="data/booking_challenge_tpp_labeled.csv",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    coldict = {
        "sort_cols": ["id", "time"],
        "event_cols": ["event"],
        "id_cols": "id",
        "scale_cols": ["time"],
        "necess_cols": ["id", "event", "time"],
    }
    #train_dataset = read_data(
    #    args.train_file, args.history_dim, args.input_dim, coldict, load=True
    #)
    #valid_dataset = read_data(
    #    args.test_file, args.history_dim, args.input_dim, coldict, load=True
    #)
    #train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    # booking
    args.train_file = 'data/booking_tensor_scaled.pt'
    full_dataset = read_data(
        args.train_file, args.history_dim, args.input_dim, coldict, load=True
    )
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    model = RNNModel(
        args.num_emb,
        args.vocab_size,
        args.input_dim,
        args.emb_dim,
        args.hidden_dim,
        args.encoder_dim,
        args.layers,
    )
    print(model)
    dataname = args.train_file.split("/")[-1]
    dataname = dataname.split(".")[0]
    writer = SummaryWriter(log_dir="runs/" + dataname + '_lstmfixed')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        model = model.cuda()

    for epoch in range(1, args.epochs + 1):
        print("training: epoch ", epoch)
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0
        event_probs = torch.zeros(batch_size, 1, args.vocab_size)
        event_count = {}
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)

        model.train()
        for inputs, _ in train_loader:

            if train_on_gpu:
                inputs = inputs.cuda()
            if len(inputs) != batch_size:
                break
            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            optimizer.zero_grad()
            cat_pred, noncat_pred, cat_logits, hidden1, hidden2 = model(
                inputs, hidden1, hidden2
            )
            loss = model.total_loss(cat_logits, noncat_pred, inputs)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)
        for inputs, _ in valid_loader:
            if train_on_gpu:
                inputs = inputs.cuda()
            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            cat_pred, noncat_pred, cat_logits, hidden1, hidden2 = model(
                inputs, hidden1, hidden2
            )
            loss = model.total_loss(cat_logits, noncat_pred, inputs)
            valid_loss += loss.item() * inputs.size(0)
            cat_targets = inputs[:, :, 0].unsqueeze_(-1)
            output, counts = torch.unique(cat_targets, return_counts=True)
            for i in range(len(output)):
                event = output[i].item()
                if event not in event_count.keys():
                    event_count[event] = counts[i].item()
                else:
                    event_count[event] += counts[i].item()

            accuracy += (cat_pred == cat_targets).float().sum().item()

        # test output
        inputs, _ = next(iter(valid_loader))
        cat_pred, noncat_pred, cat_logits, _, _ = model(inputs.cuda(), hidden1, hidden2)
        # print(inputs[-1])
        # print(torch.cat((cat_pred,noncat_pred), dim=-1)[-1])

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        # baseline and accuracy
        totalsum = 0
        mostfrequent = next(iter(event_count.values()))
        for key in event_count.keys():
            totalsum += event_count[key]
            if event_count[key] > mostfrequent:
                mostfrequent = event_count[key]

        baseline = mostfrequent / totalsum
        accuracy = accuracy / (inputs.size(0) * inputs.size(1) * len(valid_loader))
        print(
            "Epoch: {} \tTraining Loss:{:.6f} \tValidation Loss:{:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )
        print("Accuracy:{:.4f} \tBaseline:{:.4f}".format(accuracy, baseline))

        writer.add_scalars(
            "Autoencoder loss", {"train": train_loss, "validation": valid_loss}, epoch
        )
        writer.add_scalars(
            "Autoencoder accuracy",
            {"validation": accuracy, "baseline": baseline},
            epoch,
        )
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.checkpoint_dir, str(dataname) + "-checkpoint-%d.pth" % epoch
                ),
            )

    writer.close()
