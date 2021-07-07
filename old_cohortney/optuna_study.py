import optuna
import os
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
from models.LSTM import LSTMMultiplePointProcesses
from utils.file_system_utils import create_folder
import torch
import pickle
import json
import numpy as np


def objective(trial):
    
    with open("optuna_config.json", "r") as f:
        args = json.load(f)
    
    dr = trial.suggest_float(name="dropout_lstm", low=0.2, high=0.8)
    nlayers = trial.suggest_int(name="numlayers_lstm", low=1, high=5, step=1)
    hsize = trial.suggest_int(name="hiddensize_lstm", low=64, high=256, step=16)
    # Generate the model.
    model = LSTMMultiplePointProcesses(
        args["n_classes"] + 1,
        hsize,
        nlayers,
        args["n_classes"],
        args["n_clusters"],
        args["n_steps"],
        dropout=dr,#args["dropout"],
    ).to(args["device"])

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) #for hp tuning
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True) #for hp tuning
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    data, target = get_dataset(
        args["path_to_files"], args["n_classes"], args["n_steps"]
    )
    
    best_model_path = 'experiments/lstmstudy/best_model.pt'

    trainer = TrainerClusterwise(
        model,
        optimizer,
        args["device"],
        data,
        args["n_clusters"],
        exper_path='experiments/lstmstudy/',
        target=target,
        epsilon=args["epsilon"],
        max_epoch=args["max_epoch"],
        max_m_step_epoch=args["max_m_step_epoch"],
        lr=args["lr"],
        random_walking_max_epoch=args["random_walking_max_epoch"],
        true_clusters=args["true_clusters"],
        upper_bound_clusters=args["upper_bound_clusters"],
        lr_update_tol=args["lr_update_tol"],
        lr_update_param=args["lr_update_param"],
        min_lr=args["min_lr"],
        updated_lr=args["updated_lr"],
        batch_size=args["batch_size"],
        verbose=args["verbose"],
        best_model_path=best_model_path if args["save_best_model"] else None,
        max_computing_size=args["max_computing_size"],
        full_purity=args["full_purity"],
        trial=trial,
    )
    losses, results, cluster_part, stats = trainer.train()
    
    # report final obtained purity

    return results[-1][1]


if __name__ == "__main__":
    

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, catch=(RuntimeError,))

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
