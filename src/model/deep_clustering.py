import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from pathlib import Path
import pandas as pd
import json
from test_tube import Experiment

from src.networks.seq_cnn import SeqCNN
from src.networks.clustering import arrange_clustering, cluster_assign, Kmeans
from src.utils.datamodule import load_data
from src.utils.metrics import purity, consistency
from src.utils import make_grid
from src.utils.cohortney_utils import arr_func, multiclass_fws_array, events_tensor


def deep_cluster_train(config):
    args = config.aux_module
    exp_deep_cluster = Experiment(config.logger.test_tube.save_dir, config.logger.test_tube.name + '_deep_cluster' +'/'+ config.data_dir.split('/')[-1])
    exp_deep_cluster.tag({'deep_cluster': True})
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ss, _, class2idx, user_list = load_data(Path(config.data_dir), maxsize=args.maxsize, maxlen=args.maxlen,
                                            ext=args.ext, datetime=not args.not_datetime, type_=args.type)

    grid = make_grid(args.gamma, args.Tb, args.Th, args.N, args.n)
    T_j = grid[-1]
    Delta_T = np.linspace(0, grid[-1], 2 ** args.n)
    Delta_T = Delta_T[Delta_T < int(T_j)]
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
        print(f'============= RUN {run_id + 1} ===============')
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
        dataloader = DataLoader(dataset,
                                shuffle=False,
                                batch_size=config.experiment.batch_size,
                                num_workers=config.experiment.num_workers,
                                pin_memory=True)

        deepcluster = Kmeans(args.nmb_cluster)
        cluster_log = []

        for epoch in range(args.start_epoch, args.epochs):
            # remove head
            model.top_layer = None

            # get the features for the whole dataset
            features = compute_features(dataloader, model, len(dataset), config.experiment.batch_size, device=device)

            # cluster the features
            if args.verbose:
                print('Cluster the features')
            clustering_loss, I = deepcluster.cluster(features, verbose=args.verbose)

            if Path(config.data_dir, 'clusters.csv').exists():
                gt_labels = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
                gt_labels = torch.LongTensor(gt_labels)

                pur = purity(torch.LongTensor(I), gt_labels)
                exp_deep_cluster.log({'purity': pur})

                if args.verbose:
                    print(f'Purity: {pur:.4f}')

            # assign pseudo-labels
            if args.verbose:
                print('Assign pseudo labels')
            train_dataset = cluster_assign(deepcluster.lists, dataset)

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=config.experiment.batch_size,
                num_workers=config.experiment.num_workers,
                pin_memory=True,
            )

            # set last fully connected layer
            model.top_layer = nn.Linear(fd, args.nmb_cluster)  # len(deepcluster.lists))
            model.top_layer.weight.data.normal_(0, 0.01)
            model.top_layer.bias.data.zero_()
            model.top_layer.to(device)

            # train network with clusters as pseudo-labels
            end = time.time()
            loss = train(train_dataloader, model, criterion, optimizer, device, args)

            # print log
            if args.verbose:
                print(
                    f'###### Epoch {epoch} ###### \n Time: {(time.time() - end):.3f} s\n Clustering loss: {clustering_loss:.3f} \n ConvNet loss: {loss:.3f}')
                try:
                    nmi = normalized_mutual_info_score(
                        arrange_clustering(deepcluster.lists),
                        arrange_clustering(cluster_log[-1])
                    )
                    print(f'NMI against previous assignment: {nmi:.3f}')
                except IndexError:
                    pass
                print('####################### \n')
            cluster_log.append(deepcluster.lists)

        assigned_labels.append(I)
        if args.verbose:
            print(
                f'Sizes of clusters: {", ".join([str((torch.tensor(I) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')

    assigned_labels = torch.LongTensor(assigned_labels)
    cons = consistency(assigned_labels)

    print(assigned_labels)
    if args.verbose:
        print(f'Consistncy: {cons}\n')

    results = {'consistency': cons}

    if Path(args.data_dir, 'clusters.csv').exists():
        gt_labels = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_labels = torch.LongTensor(gt_labels)

        pur_val_mean = np.mean([purity(x, gt_labels) for x in assigned_labels])
        pur_val_std = np.std([purity(x, gt_labels) for x in assigned_labels])

        print(f'\nPurity: {pur_val_mean}+-{pur_val_std}')

        results['purity'] = (pur_val_mean, pur_val_std)

    exp_deep_cluster.save()
    exp_deep_cluster.close()

    if args.result_path is not None:
        json.dump(results, Path(f'{args.result_path}.json'))


def train(loader, model, crit, opt, device, args=None):
    """
    Training of the CNN.
    Args:
        loader (torch.utils.data.DataLoader): Data loader
        model (nn.Module): CNN
        crit (torch.nn): loss
        opt (torch.optim.SGD): optimizer for every parameters with True
                               requires_grad in model except top layer
    """
    total_loss = 0
    N = 0
    model.train()

    # create an optimizer for the last fc layer
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
def compute_features(dataloader, model, N, batch_size, device):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        input_var = torch.autograd.Variable(input_tensor)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux
        else:
            # special treatment for final batch
            features[i * batch_size:] = aux

    return features
