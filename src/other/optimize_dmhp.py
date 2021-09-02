from pathlib import Path
import argparse
import torch
import numpy as np
import time

from models import DirichletMixtureModel, PointProcess, EMClustering
from utils.metrics import consistency, purity, update_info_score
from utils.preprocessing import load_data_dmhp
import sys
import json


def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="dir holding sequences as separate files",
    )
    parser.add_argument(
        "--nmb_cluster", type=int, default=10, help="number of clusters"
    )
    # hyperparameters for Cohortney
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--Tb", type=float, default=7e-6)
    parser.add_argument("--Th", type=float, default=80)
    parser.add_argument("--N", type=int, default=2500)
    parser.add_argument("--n", type=int, default=8, help="n for partition")
    # hyperparameters for training
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--save_to",
        type=str,
        default="DMHP_Metrics",
        help="directory for saving metrics",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for dataloader"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--nruns", type=int, default=3, help="number of trials")
    parser.add_argument("--type", type=str, default=None, help="if it is a")

    parser.add_argument("--result_path", type=str, help="path to save results")
    args = parser.parse_args()
    return args


sys.path.append("..")

if __name__ == "__main__":

    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ss, Ts, class2idx, gt_ids = load_data_dmhp(Path(args.data_dir))
    # N = len(ss)

    hawkes_process = PointProcess(ss, Ts, eps=1e5, tune=False)
    # the dimension of Hawkes processes
    D = len(hawkes_process.basis_fs)
    C = len(class2idx)
    K = args.nmb_cluster
    # parameter for inner iterations of EM
    niter = 10
    labels = torch.zeros(args.nruns, len(ss))
    info_score = np.zeros((K + 1, K + 1))
    nlls = torch.zeros(args.nruns, niter)
    times = np.zeros(args.nruns)

    assigned_labels = []
    results = {}
    for i in range(args.nruns):
        print(f"============= RUN {i+1} ===============")
        time_start = time.time()
        dirichlet_mixture_model = DirichletMixtureModel(K, C, D)
        print("Dirichlet Mixture Model is initiated")
        EM = EMClustering(hawkes_process, dirichlet_mixture_model)
        print("EM clustering model is initiated")
        r, nll_history, r_history = EM.learn_hp(
            niter=niter, ninner=[2, 3, 4, 5, 6, 7] + (niter - 6) * [8]
        )
        print("Training is completed")

        labels[i] = r.argmax(-1)
        assigned_labels.append(labels[i])
        # print("predicted labels:", labels[i])
        nlls[i] = torch.FloatTensor(nll_history)

        # if args.verbose:
        #    print(
        #        f'Sizes of clusters: {", ".join([str((torch.tensor(labels[i]) == i).sum().item()) for i in range(args.nmb_cluster)])}\n'
        #    )

        if gt_ids is not None:
            print(f"Purity: {purity(labels[i], gt_ids):.4f}")
            info_score = update_info_score(info_score, labels[i], gt_ids, K, args.nruns)

        times[i] = time.time() - time_start

    cons = consistency(assigned_labels)
    print(f"Consistency: {cons:.4f}\n")
    results["consistency"] = cons
    if gt_ids is not None:
        purity_mean = np.mean([purity(x, gt_ids) for x in labels])
        purity_std = np.std([purity(x, gt_ids) for x in labels])
        print(f"Purity: {purity_mean:.4f}+-{purity_std:.4f}")
        print(f"Normalized mutual info score: {info_score}")
    time_mean = np.mean(times)
    time_std = np.std(times)
    nll_mean = torch.mean(nlls)
    nll_std = torch.std(nlls)
    print(f"Mean run time: {time_mean:.4f}+-{time_std:.4f}")
    if (args.save_to is not None) and (gt_ids is not None):
        metrics = {
            "Purity": f"{purity_mean:.4f}+-{purity_std:.4f}",
            "Mean run time": f"{time_mean:.4f}+-{time_std:.4f}",
            "Normalized mutual info score:": f"{info_score}",
            "Predictive log likelihood:": f"{nll_mean.item():.4f}+-{nll_std.item():.4f}",
            "Predicted labels": f"{labels}",
        }
        # don't think it's a good idea to dump labels together with metrics
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        metrics = {
            "Mean run time": f"{time_mean:.4f}+-{time_std:.4f}",
            "Predictive log likelihood:": f"{nll_mean.item():.4f}+-{nll_std.item():.4f}",
            "Predicted labels": f"{labels}",
        }
        # don't think it's a good idea to dump labels together with metrics
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
