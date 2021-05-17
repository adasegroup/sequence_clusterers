import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam
import numpy as np
import argparse
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from pathlib import Path
import itertools
import pandas as pd
import json

from models.cnn.model import SeqCNN
import src.DeepCluster.clustering as clustering
from src.Cohortney.data_utils import load_data, sep_hawkes_proc
from src.Cohortney.utils import make_grid, purity, consistency
from src.Cohortney.cohortney import arr_func, multiclass_fws_array, events_tensor


def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--maxsize', type=int, default=None, help='max number of sequences')
    parser.add_argument('--nmb_cluster', type=int, default=10, help='number of clusters')
    parser.add_argument('--maxlen', type=int, default=-1, help='maximum length of sequence')
    parser.add_argument('--ext', type=str, default='txt', help='extention of files with sequences')
    parser.add_argument('--not_datetime', action='store_true', help='if time values in event sequences are represented in datetime format')
    # hyperparameters for Cohortney
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Tb', type=float, default=7e-6)
    parser.add_argument('--Th', type=float, default=80)
    parser.add_argument('--N', type=int, default=2500)
    parser.add_argument('--n', type=int, default=4, help='n for partition')
    # hyperparameters for training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nruns', type=int, default=1, help='number of trials')
    parser.add_argument('--type', type=str, default=None, help='if it is booking data or not')

    parser.add_argument('--result_path', type=str, help='path to save results')
    args = parser.parse_args()
    with open('commandline_args2.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args

np.set_printoptions(threshold=10000)
torch.set_printoptions(threshold=10000)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') # remove it!
    ss, _, class2idx, user_list = load_data(Path(args.data_dir), maxsize=args.maxsize, maxlen=args.maxlen, ext=args.ext, datetime=not args.not_datetime, type_=args.type)
    
    grid = make_grid(args.gamma, args.Tb, args.Th, args.N, args.n)
    T_j = grid[-1]
    Delta_T = np.linspace(0, grid[-1], 2**args.n)
    Delta_T = Delta_T[Delta_T< int(T_j)]
    Delta_T = tuple(Delta_T)

    _, events_fws_mc = arr_func(user_list, T_j, Delta_T, multiclass_fws_array)
    mc_batch = events_tensor(events_fws_mc)
    dataset = torch.FloatTensor(mc_batch)

    if args.verbose:
        print('Loaded data')
        print(f'Dataset shape: {list(dataset.shape)}')
    input_size = dataset.shape[-1]

    assigned_labels = []
    for run_id in range(args.nruns):
        print(f'============= RUN {run_id+1} ===============')
        in_channels = len(class2idx)
        model = SeqCNN(input_size, in_channels, device=device)
        model.top_layer = None
        model.to(device)
        fd = model.fd
        
        optimizer = Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
        criterion = nn.CrossEntropyLoss()

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                pin_memory=True)

        deepcluster = clustering.Kmeans(args.nmb_cluster)
        cluster_log = []

        for epoch in range(args.start_epoch, args.epochs):
            end = time.time()

            # remove head
            model.top_layer = None

            # get the features for the whole dataset
            features = compute_features(dataloader, model, len(dataset), device)

            # cluster the features
            if args.verbose:
                print('Cluster the features')
            clustering_loss, I = deepcluster.cluster(features, verbose=args.verbose)

            if Path(args.data_dir, 'clusters.csv').exists() and args.verbose:
                gt_labels = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
                gt_labels = torch.LongTensor(gt_labels)
                
                pur = purity(torch.LongTensor(I), gt_labels)
                if args.verbose:
                    print(f'Purity: {pur:.4f}')

            # assign pseudo-labels
            if args.verbose:
                print('Assign pseudo labels')
            train_dataset = clustering.cluster_assign(deepcluster.lists,
                                                    dataset)

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args.batch,
                num_workers=args.workers,
                pin_memory=True,
            )

            # set last fully connected layer
            model.top_layer = nn.Linear(fd, args.nmb_cluster) #len(deepcluster.lists))
            model.top_layer.weight.data.normal_(0, 0.01)
            model.top_layer.bias.data.zero_()
            model.top_layer.to(device)

            # train network with clusters as pseudo-labels
            end = time.time()
            loss = train(train_dataloader, model, criterion, optimizer, epoch, device)

            # print log
            if args.verbose:
                print(f'###### Epoch {epoch} ###### \n Time: {(time.time() - end):.3f} s\n Clustering loss: {clustering_loss:.3f} \n ConvNet loss: {loss:.3f}')
                try:
                    nmi = normalized_mutual_info_score(
                        clustering.arrange_clustering(deepcluster.lists),
                        clustering.arrange_clustering(cluster_log[-1])
                    )
                    print(f'NMI against previous assignment: {nmi:.3f}')
                except IndexError:
                    pass
                print('####################### \n')
            cluster_log.append(deepcluster.lists)

        assigned_labels.append(I)
        if args.verbose:
            print(f'Sizes of clusters: {", ".join([str((torch.tensor(I) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')
    assigned_labels = torch.LongTensor(assigned_labels)
    cons = consistency(assigned_labels)
    
    print (assigned_labels)
    if args.verbose:
        print(f'Consistency: {cons}\n')

    results = {'consistency': cons}

    if Path(args.data_dir, 'clusters.csv').exists():
        gt_labels = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_labels = torch.LongTensor(gt_labels)
        
        pur_val_mean = np.mean([purity(x, gt_labels) for x in assigned_labels])
        pur_val_std = np.std([purity(x, gt_labels) for x in assigned_labels])

        print(f'\nPurity: {pur_val_mean}+-{pur_val_std}')

        results['purity'] = (pur_val_mean, pur_val_std)

    if args.result_path is not None:
        json.dump(results, Path(f'{args.result_path}.json'))


def train(loader, model, crit, opt, epoch, device):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    total_loss = 0
    N = 0
    model.train()

    #create an optimizer for the last fc layer
    optimizer_tl = SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        momentum=args.momentum,
    )

    for i, (input_tensor, target) in enumerate(loader):
        target = target.to(device)
        input_tensor = input_tensor.to(device)

        output = model(input_tensor)
        loss = crit(output, target)

        total_loss += loss.item()
        N += input_tensor.shape[0]

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

    avg_loss = total_loss / N

    return avg_loss

@torch.no_grad()
def compute_features(dataloader, model, N, device):
    if args.verbose:
        print('Compute features')
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)
