import torch

def log_likelihood_single(partitions, lambdas, dts):
    tmp1 = lambdas*dts[:, None, None]
    p = partitions[:,:,1:]
    return torch.sum(tmp1) - torch.sum(p*torch.log(tmp1))

def log_likelihood(partitions, lambdas, dts, mixing, gamma):
    dts = dts[None,:,None, None]
    tmp = lambdas*dts
    p = partitions[None,:,:,1:]
    return torch.sum(gamma*torch.sum(tmp1 - p*torch.log(tmp1), dim = (2,3)))

def purity(learned_ids, gt_ids):
    """
    Args:
    - learned_ids - 1-D tensor of labels obtained from model
    - gt_ids - 1-D tensor of ground truth labels
    """
    assert len(learned_ids) == len(gt_ids)
    pur = 0
    ks = torch.unique(learned_ids)
    js = torch.unique(gt_ids)
    for k in ks:
        inters = []
        for j in js:
            inters.append(((learned_ids == k) * (gt_ids == j)).sum().item())
        pur += 1./len(learned_ids) * max(inters)

    return pur
