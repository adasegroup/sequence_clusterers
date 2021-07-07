import torch
import time
import numpy as np
from utils.metrics import purity
import math
from models.LSTM import LSTMMultiplePointProcesses
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
import pickle
import json
import os
import pandas as pd
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="booking")
    parser.add_argument("--col_to_select", type=str, default=None)
    parser.add_argument("--experiment_n", type=str, default="exp_0")
    args = parser.parse_args()
    # path to dataset
    datapath = os.path.join("data", args.dataset)
    # path to experiment settings and weights
    experpath = os.path.join("experiments", args.dataset)
    experpath = os.path.join(experpath, args.experiment_n)
    modelweights = os.path.join(experpath, "last_model.pt")
    with open(os.path.join(experpath, "args.json")) as json_file:
        config = json.load(json_file)
    n_steps = config["n_steps"]
    n_classes = config["n_classes"]
    # init model
    model = LSTMMultiplePointProcesses(
        n_classes + 1,
        config["hidden_size"],
        config["num_layers"],
        n_classes,
        config["n_clusters"],
        n_steps,
        dropout=config["dropout"],
    ).to(config["device"])
    optimizer = torch.optim.Adam(
        model.parameters(), config["lr"], weight_decay=config["weight_decay"]
    )
    model = torch.load(modelweights, map_location=torch.device(config["device"]))
    model.eval()
    # start
    start_time = time.time()
    data, target = get_dataset(datapath, model.num_classes, n_steps, args.col_to_select)

    trainer = TrainerClusterwise(
        model,
        optimizer,
        config["device"],
        data,
        model.num_clusters,
        exper_path=experpath,
        target=target,
        epsilon=config["epsilon"],
        max_epoch=config["max_epoch"],
        max_m_step_epoch=config["max_m_step_epoch"],
        lr=config["lr"],
        random_walking_max_epoch=config["random_walking_max_epoch"],
        true_clusters=config["true_clusters"],
        upper_bound_clusters=config["upper_bound_clusters"],
        lr_update_tol=config["lr_update_tol"],
        lr_update_param=config["lr_update_param"],
        min_lr=config["min_lr"],
        updated_lr=config["updated_lr"],
        batch_size=config["batch_size"],
        verbose=config["verbose"],
        best_model_path=None,
        max_computing_size=config["max_computing_size"],
        full_purity=config["full_purity"],
    )

    # infer clusters
    if trainer.max_computing_size is None:
        lambdas = trainer.model(trainer.X.to(config["device"]))
        trainer.gamma = trainer.compute_gamma(lambdas)
        clusters = torch.argmax(trainer.gamma, dim=0)
    else:
        # large datasets
        trainer_data = trainer.X
        lenX = len(trainer.X)
        for i in range(0, lenX // trainer.max_computing_size + 1):
            trainer.X = trainer_data[
                i * trainer.max_computing_size : (i + 1) * trainer.max_computing_size
            ]
            lambdas = trainer.model(trainer.X.to(config["device"]))
            trainer.gamma = torch.zeros(trainer.n_clusters, len(trainer.X)).to(
                config["device"]
            )
            trainer.gamma = trainer.compute_gamma(lambdas)
            if i == 0:
                clusters = torch.argmax(trainer.gamma, dim=0)
            else:
                currclusters = torch.argmax(trainer.gamma, dim=0)
                clusters = torch.cat((clusters, currclusters), dim=0)

    end_time = time.time()
    # save results
    res_df = pd.read_csv(os.path.join(datapath, "clusters.csv"))
    res_df["time"] = round(end_time - start_time, 5)
    res_df["seqlength"] = 0
    csvfiles = sorted(os.listdir(datapath))
    for index, row in res_df.iterrows():
        seq_df = pd.read_csv(os.path.join(datapath, csvfiles[index]))
        res_df.at[index, "seqlength"] = len(seq_df)

    res_df["coh_cluster"] = clusters.detach().cpu().numpy().tolist()
    savepath = os.path.join(experpath, "inferredclusters.csv")
    res_df.drop(
        res_df.columns[res_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    res_df.to_csv(savepath)
