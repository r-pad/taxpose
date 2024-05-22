import typing
from typing import Callable, List, Optional

import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.pointnet import PointNet


@typing.no_type_check
def estimate_p(
    P: torch.FloatTensor, R: torch.FloatTensor, W: Optional[torch.FloatTensor] = None
) -> torch.FloatStorage:
    assert P.ndim == 3  # N x D x 1
    assert R.ndim == 1  # N
    assert P.shape[0] == R.shape[0]
    assert P.shape[1] in {2, 3}

    N, D, _ = P.shape

    if W is None:
        W = torch.ones(N, device=P.device)
    assert W.ndim == 1  # N
    W = W[:, None, None]

    # Shared stuff.
    Pt = P.permute(0, 2, 1)
    PPt = P @ Pt
    PtP = (Pt @ P).squeeze()
    I = torch.eye(D, device=P.device)
    NI = I[None].repeat(N, 1, 1)
    PtP_minus_r2 = (PtP - R**2)[:, None, None]

    # These are ripped straight from the paper, with weighting passed through.
    a = (W * (PtP_minus_r2 * P)).mean(dim=0)
    B = (W * (-2 * PPt - PtP_minus_r2 * NI)).mean(dim=0)
    c = (W * P).mean(dim=0)
    f = a + B @ c + 2 * c @ c.T @ c
    H = -2 * PPt.mean(dim=0) + 2 * c @ c.T
    q = -torch.linalg.inv(H) @ f
    p = q + c

    return p


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)


class MLPKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = MLP(2 * feature_dim, [300, 100, 1])

    def forward(self, x1, x2):
        # Make it symmetric.
        # b = torch.stack(
        #     [
        #         torch.cat([x1, x2], axis=-1),
        #         torch.cat([x2, x1], axis=-1),
        #     ],
        #     axis=0,
        # )
        v1 = self.mlp(torch.cat([x1, x2], axis=-1))
        v2 = self.mlp(torch.cat([x2, x1], axis=-1))
        return F.softplus((v1 + v2) / 2)


class NormKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x1, x2):
        return torch.norm(x1 - x2, dim=-1) / math.sqrt(len(x1))


class DotProductKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x1, x2):
        return torch.dot(x1, x2) / math.sqrt(len(x1))


class MultilaterationHead(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        n_kps=100,
        pred_weight=True,
        last_attn=False,
        sample: bool = False,
        pred_mlat_weight: bool = False,
    ):
        super().__init__()

        self.emb_dims = emb_dims
        self.n_kps = n_kps
        self.last_attn = last_attn

        self.kernel = MLPKernel(self.emb_dims - int(last_attn))
        # self.kernel = NormKernel(self.emb_dims)
        self.sample = sample

        self.pred_mlat_weight = pred_mlat_weight
        if self.pred_mlat_weight:
            self.proj_mlat_weight = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims - int(last_attn), 64, 64, 64, 128, 512]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

    def forward(
        self,
        *input,
        scores=None,
        return_flow_component=False,
        return_embedding=False,
        conditioning=None,
    ):
        action_embedding = input[0]
        anchor_embedding = input[1]

        if self.last_attn:
            action_embedding, action_attn = (
                action_embedding[:, :-1],
                action_embedding[:, -1:],
            )
            anchor_embedding, anchor_attn = (
                anchor_embedding[:, :-1],
                anchor_embedding[:, -1:],
            )

        action_points = input[2]
        anchor_points = input[3]

        P_A = action_points.permute(0, 2, 1)
        P_B = anchor_points.permute(0, 2, 1)

        Phi_A = action_embedding.permute(0, 2, 1)
        Phi_B = anchor_embedding.permute(0, 2, 1)

        if self.last_attn:
            A_weights = action_attn.permute(0, 2, 1)
            B_weights = anchor_attn.permute(0, 2, 1)

            A_weights = F.softmax(A_weights, dim=-1).squeeze(dim=-1)
            B_weights = F.softmax(B_weights, dim=-1).squeeze(dim=-1)

            # Should sum to N.
            A_weights = A_weights * A_weights.shape[-1]
            B_weights = B_weights * B_weights.shape[-1]
        elif self.pred_mlat_weight:
            # TODO: Fix this
            A_weights = self.proj_mlat_weight(action_embedding).squeeze(dim=1)
            B_weights = self.proj_mlat_weight(anchor_embedding).squeeze(dim=1)

            A_weights = F.softmax(A_weights, dim=-1)
            B_weights = F.softmax(B_weights, dim=-1)

            A_weights = A_weights * A_weights.shape[-1]
            B_weights = B_weights * B_weights.shape[-1]

        else:
            A_weights = torch.ones(Phi_A.shape[:2], device=Phi_A.device)
            B_weights = torch.ones(Phi_B.shape[:2], device=Phi_B.device)

        # We probably want to sample
        if self.sample:
            # This function samples without replacement, in a batch.
            choice_v = functorch.vmap(
                lambda x, n: torch.randperm(x.shape[-1])[:n],
                in_dims=(0, None),
                randomness="different",
            )
            A_ixs = choice_v(action_points, self.n_kps).to(action_points.device)
            B_ixs = choice_v(anchor_points, self.n_kps).to(anchor_points.device)
            P_A = torch.take_along_dim(P_A, A_ixs.unsqueeze(-1), dim=1)
            Phi_A = torch.take_along_dim(Phi_A, A_ixs.unsqueeze(-1), dim=1)
            P_B = torch.take_along_dim(P_B, B_ixs.unsqueeze(-1), dim=1)
            Phi_B = torch.take_along_dim(Phi_B, B_ixs.unsqueeze(-1), dim=1)
            A_weights = torch.take_along_dim(A_weights, A_ixs, dim=1)
            B_weights = torch.take_along_dim(B_weights, B_ixs, dim=1)
        else:
            bs = P_A.shape[0]
            A_ixs = torch.arange(P_A.shape[1], device=P_A.device).repeat(bs, 1)
            B_ixs = torch.arange(P_B.shape[1], device=P_B.device).repeat(bs, 1)

        # compute_R = functorch.vmap(
        #     functorch.vmap(
        #         functorch.vmap(self.kernel, in_dims=(None, 0)), in_dims=(0, None)
        #     ),
        #     in_dims=(0, 0),
        # )
        # R_est = compute_R(Phi_A, Phi_B)
        Phi_A_r = (
            Phi_A.unsqueeze(2)
            .repeat(1, 1, Phi_A.shape[1], 1)
            .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_A.shape[1], Phi_A.shape[2])
        )
        Phi_B_r = (
            Phi_B.unsqueeze(1)
            .repeat(1, Phi_B.shape[1], 1, 1)
            .reshape(Phi_B.shape[0] * Phi_B.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
        )
        R_est = self.kernel(Phi_A_r, Phi_B_r).reshape(
            Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1]
        )

        # R_est = torch.cdist(Phi_A, Phi_B, p=2.0) / math.sqrt(self.emb_dims)

        # Normalize the scores.
        # mlat_weights = (
        #     scores / scores.detach().sum(dim=-1, keepdim=True) * scores.shape[-1]
        # )
        # mlat_weights = torch.ones_like(scores, device=scores.device)
        v_est_p = functorch.vmap(functorch.vmap(estimate_p, in_dims=(None, 0, None)))
        P_A_B_pred = v_est_p(P_B[..., None], R_est, B_weights)[..., 0]

        corr_points = P_A_B_pred.permute(0, 2, 1)
        flow = (P_A_B_pred - P_A).permute(0, 2, 1)

        # TODO: figure out how to downsample the points, and pass it all back up the stack.

        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        # flow = corr_points - action_points

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            if self.sample:
                weight = torch.take_along_dim(weight, A_ixs.unsqueeze(1), dim=2)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

        return {
            "full_flow": corr_flow_weight,
            "residual_flow": torch.zeros_like(flow).to(flow.device),
            "corr_flow": flow,
            "corr_points": corr_points,
            "scores": scores,
            "P_A": P_A.permute(0, 2, 1),
            "A_ixs": A_ixs,
        }
