import torch


def loss_getter(model_type: str):
    """
    Returns the criterion to corresponding model
    """
    if model_type == "LSTM":
        return cohortney_log_loss
    else:
        raise Exception("Unknown model type")


def cohortney_log_loss(
        partitions: torch.Tensor,  # shape = [batch_size, n_steps, n_classes + 1]
        lambdas: torch.Tensor,  # shape = [n_clusters, batch_size, n_steps, n_classes]
        gamma: torch.Tensor,  # shape = [n_clusters, batch_size]
        epsilon: float
):
    """
    Computes negative log likelihood for partitions
    """
    # computing poisson parameters
    dts = partitions[:, 0, 0]  # shape = [batch_size,]
    dts = dts[None, :, None, None]  # shape = [1, batch_size, 1, 1]
    poisson_param = lambdas * dts  # shape = [n_clusters, batch_size, n_steps, n_classes]

    # preparing partitions
    p = partitions[None, :, :, 1:]  # shape = [1, batch_size, n_steps, n_classes]

    # computing negative log likelihoods of every timestamp
    timestamp_nll = poisson_param - p * torch.log(poisson_param + epsilon) + torch.lgamma(p + 1)

    # computing log likelihoods of data points
    cluster_batch_nll = torch.sum(timestamp_nll, dim=(2, 3))

    # computing loss
    em_loss = gamma * cluster_batch_nll
    em_loss = torch.sum(em_loss)

    return em_loss
