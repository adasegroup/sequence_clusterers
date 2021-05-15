import torch


def cohortney_criterion(partitions: torch.Tensor, lambdas: torch.Tensor, gamma: torch.Tensor, epsilon=1e-6):
    """
        Computes loss
        :arg:
            partitions - torch.Tensor, size = (batch_size, seq_len, number of classes + 1)
            lambdas - torch.Tensor, size = (batch_size, seq_len, number of classes), model output
            gamma - torch.Tensor, size = (n_clusters, batch_size), probabilities p(k|x_n)
        :param:
            loss - torch.Tensor, size = (1), sum of output log likelihood weighted with convoluted gamma
                   and prior distribution log likelihood
    """
    # computing poisson parameters
    dts = partitions[:, 0, 0]
    dts = dts[None, :, None, None]
    tmp = lambdas * dts

    # preparing partitions
    p = partitions[None, :, :, 1:]

    # computing log likelihoods of every timestamp
    tmp1 = tmp - p * torch.log(tmp + epsilon) + torch.lgamma(p + 1)

    # computing log likelihoods of data points
    tmp2 = torch.sum(tmp1, dim=(2, 3))

    # computing loss
    tmp3 = gamma * tmp2
    loss = torch.sum(tmp3)
    return loss