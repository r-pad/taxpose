import numpy as np
import torch
import torch.nn.functional as F


def js_div(p, q, reduction="batchmean", eps=1e-8):
    """
    Calculate Jensen-Shannon divergence JSD(p||q) for each sample in the batch.

    Args:
        p: Tensor of shape (batch_size, num_classes) containing class probabilities.
        q: Tensor of shape (batch_size, num_classes) containing class probabilities.
        reduction: Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'.
        eps: Small constant to add to the denominator.
    """
    m = 0.5 * (p.exp() + q.exp())

    m = torch.clamp(m + eps, eps)

    jsd = 0.5 * F.kl_div(
        m.log(), p, log_target=True, reduction="none"
    ) + 0.5 * F.kl_div(m.log(), q, log_target=True, reduction="none")

    if reduction == "mean":
        jsd = jsd.mean()
    elif reduction == "sum":
        jsd = jsd.sum()
    elif reduction == "batchmean":
        jsd = jsd.sum() / p.size(0)
    else:
        pass

    return jsd


def js_div_mod(p, q, d_1=0.01, d_2=0.1, reduction="batchmean", eps=1e-8):
    """
    Calculate 0.5( KL(p+d_1||q+d_1) + KL(q+d_2||p+d_2) ) for each sample in the batch.))

    Args:
        p: Tensor of shape (batch_size, num_classes) containing class log-probabilities.
        q: Tensor of shape (batch_size, num_classes) containing class log-probabilities.
        d_1: Small constant to add to p and q.
        d_2: Small constant to add to p and q.
        reduction: Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'.
        eps: Small constant to add to the denominator.
    """

    p1_ = torch.clamp(p.exp() + d_1 + eps, eps)
    p2_ = torch.clamp(p.exp() + d_2 + eps, eps)
    q1_ = torch.clamp(q.exp() + d_1 + eps, eps)
    q2_ = torch.clamp(q.exp() + d_2 + eps, eps)

    jsd = 0.5 * F.kl_div(
        q1_.log(), p1_.log(), log_target=True, reduction="none"
    ) + 0.5 * F.kl_div(p2_.log(), q2_.log(), log_target=True, reduction="none")

    if reduction == "mean":
        jsd = jsd.mean()
    elif reduction == "sum":
        jsd = jsd.sum()
    elif reduction == "batchmean":
        jsd = jsd.sum() / p.size(0)
    else:
        pass

    return jsd


def wasserstein_distance(p, q, reduction="batchmean", p_exp=1, eps=1e-8):
    """
    Calculates the 1D Wasserstein distance between two distributions.

    Args:
        p: Tensor of shape (batch_size, num_classes) containing class probabilities.
        q: Tensor of shape (batch_size, num_classes) containing class probabilities.
        reduction: Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'.
        p_exp: Exponent for the p-norm.
    """

    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)

    if p_exp == 1:
        wass = torch.abs(cdf_p - cdf_q).sum(dim=-1)
    else:
        wass = torch.pow(torch.abs(cdf_p - cdf_q), p_exp).sum(dim=-1)
        wass = torch.pow(wass, 1 / p_exp)

    if reduction == "mean":
        wass = wass.mean()
    elif reduction == "sum":
        wass = wass.sum()
    elif reduction == "batchmean":
        wass = wass.sum() / p.size(0)
    else:
        pass

    return wass
