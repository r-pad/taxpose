from typing import List, Tuple

import numpy as np
import torch


def gumbel_softmax_topk(
    logits: torch.Tensor, k: int, tau: float, hard: bool = False, dim: int = -1
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Sample the top-k largest entries from a gumbel softmax distribution
    Code based on: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html

    Args:
        logits (torch.Tensor): input tensor
        k (int): number of largest entries to sample
        tau (float): temperature parameter
        hard (bool): if True, return one-hot encoding, else return k-hot encoding
        dim (int): dimension to sample from
    Returns:
        khot (torch.Tensor): k-hot encoding
        onehot_list (torch.Tensor): if hard, return list of one-hot encodings, else None
    """
    gumbel = torch.distributions.gumbel.Gumbel(
        torch.zeros_like(logits), torch.ones_like(logits)
    )
    g = gumbel.sample()
    y = logits + g

    khot = torch.zeros_like(y)
    onehot_approx = torch.zeros_like(y)
    for i in range(k):
        khot_mask = torch.max(
            1.0 - onehot_approx,
            torch.tensor([np.finfo(np.float32).tiny], device=y.device),
        )
        y = y + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(y / tau, dim=dim)
        khot = khot + onehot_approx

    if hard:
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=dim)
        khot_hard = khot_hard.scatter_(dim, ind, 1.0)
        res = khot_hard - khot.detach() + khot

        onehot_list = []
        for i in range(k):
            onehot = torch.zeros_like(y)
            onehot = onehot.scatter_(dim, ind[:, i].unsqueeze(dim), 1.0)
            onehot_list.append(onehot)
    else:
        res = khot
        onehot_list = None

    return res, onehot_list
