from typing import Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import Rotate, random_rotations
from torch import nn
from torch.nn import functional as F


def dense_cos_similarity(psi, phi):
    phi = F.normalize(phi, dim=1)
    psi = F.normalize(psi, dim=1)
    return phi.transpose(-1, -2) @ psi


def InfoNCETraining(model, points, transform, temperature=0.1):
    points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3
    phi = model(points_centered.transpose(1, 2))  # B, emb_dim, num_points

    points_trans = transform.transform_points(points)
    points_trans_centered = points_trans - points_trans.mean(
        dim=1, keepdims=True
    )  # B, num_points, 3
    phi_trans = model(points_trans_centered.transpose(1, 2))  # B, emb_dim, num_points

    similarity = phi.transpose(-1, -2) @ phi_trans

    target = torch.arange(similarity.shape[-1], device=similarity.device).tile(
        [similarity.shape[0], 1]
    )  # B, num_points

    loss = F.cross_entropy(similarity / temperature, target)

    return loss, similarity, phi, phi_trans


def neg_cos_sim(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return -(p * z).sum(dim=1).mean()


def SimSiamLoss(model, pred_model, points, transform):
    points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3
    z = model(points_centered.transpose(1, 2))  # B, emb_dim, num_points
    p = pred_model(z)

    points_trans = transform.transform_points(points)
    points_trans_centered = points_trans - points_trans.mean(
        dim=1, keepdims=True
    )  # B, num_points, 3

    z_trans = model(points_trans_centered.transpose(1, 2)).detach()
    p_trans = pred_model(z_trans)

    loss = neg_cos_sim(p, z_trans) / 2 + neg_cos_sim(p_trans, z) / 2

    return loss, z, z_trans


def dist2mask(xyz, radius=0.02):
    d = (xyz.unsqueeze(1) - xyz.unsqueeze(2)).norm(dim=-1)
    w = (d > radius).float()
    w = w + torch.eye(d.shape[-1], device=d.device).unsqueeze(0).tile(
        [d.shape[0], 1, 1]
    )
    return w


def dist2weight(xyz, func=None):
    d = (xyz.unsqueeze(1) - xyz.unsqueeze(2)).norm(dim=-1)
    if func is not None:
        d = func(d)
    w = d / d.max(dim=-1, keepdims=True)[0]
    w = w + torch.eye(d.shape[-1], device=d.device).unsqueeze(0).tile(
        [d.shape[0], 1, 1]
    )
    return w


def infonce_loss(phi, phi_trans, weights=None, temperature=0.1):
    B, D, N = phi.shape
    similarity = phi.transpose(-1, -2) @ phi_trans
    target = torch.arange(N, device=similarity.device).tile([B, 1])
    if weights is None:
        weights = 1.0
    loss = F.cross_entropy(weights * (similarity / temperature), target)

    return loss, similarity


def mean_order(similarity):
    order = (similarity > similarity.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1)
    return order.float().mean() / similarity.shape[-1]


def mean_geo_diff(similarity, points):
    similarity_argmax = torch.argmax(similarity, dim=-1)  # B,num_points
    indices = similarity_argmax.unsqueeze(-1)  # B,num_points, 1
    indices = indices.repeat(1, 1, 3)  # B, num_points, 3
    most_similar_point = torch.gather(points, 1, indices)  # B,num_points, 3
    geo_diff = torch.norm(points - most_similar_point, dim=-1)

    return geo_diff.mean()


def compute_infonce_loss(
    model: nn.Module,
    points: torch.Tensor,
    normalize_features: bool = True,
    weighting: str = "dist",
    l2_reg_weight: float = 0.0,
    center=None,
    log_prefix: str = "",
) -> Tuple[torch.Tensor, Dict]:
    """
    Computes the InfoNCE loss for a model given a batch of points.
    https://arxiv.org/pdf/2211.09325.pdf equation 19

    Args:
        model: the model to compute the loss for
        points: the batch of points to compute the loss for, [B, N, D]
        normalize_features: whether to normalize the features before computing the loss
        weighting: the weighting scheme to use for the loss
        l2_reg_weight: the weight of the l2 regularization term
        log_prefix: the prefix to use for logging

    Returns:
        contrastive_loss: the InfoNCE loss
        log_values: a dictionary of values to log
    """
    rotations = Rotate(random_rotations(points.shape[0], device=points.device))

    if center is None:
        points_centered_xyz = points[:, :, :3] - points[:, :, :3].mean(
            dim=1, keepdims=True
        )
        points_centered = torch.cat([points_centered_xyz, points[:, :, 3:]], dim=-1)

        points_action_rotated_xyz = rotations.transform_points(points_centered_xyz)
        points_action_rotated_centered_xyz = (
            points_action_rotated_xyz
            - points_action_rotated_xyz.mean(dim=1, keepdims=True)
        )
        points_action_rotated_centered = torch.cat(
            [points_action_rotated_centered_xyz, points[:, :, 3:]], dim=-1
        )
    else:
        points_centered_xyz = points[:, :, :3] - center
        points_centered = torch.cat([points_centered_xyz, points[:, :, 3:]], dim=-1)

        points_action_rotated_xyz = rotations.transform_points(points_centered_xyz)
        center_rotated = rotations.transform_points(center)
        points_action_rotated_centered_xyz = points_action_rotated_xyz - center_rotated
        points_action_rotated_centered = torch.cat(
            [points_action_rotated_centered_xyz, points[:, :, 3:]], dim=-1
        )

    phi = model(points_centered.transpose(-1, -2))
    phi_rotated = model(points_action_rotated_centered.transpose(-1, -2))

    if normalize_features:
        phi = F.normalize(phi, dim=1)
        phi_rotated = F.normalize(phi_rotated, dim=1)

    if weighting.lower() == "mask":
        w = dist2weight(points_centered)
    elif weighting.lower() == "dist":
        w = dist2weight(points, func=lambda x: torch.tanh(10 * x))
    else:
        w = None

    contrastive_loss, similarity = infonce_loss(
        phi, phi_rotated, weights=w, temperature=0.1
    )

    mean_order_error = mean_order(similarity)
    mean_geo_error = mean_geo_diff(similarity, points[:, :, :3])

    log_values = {}
    log_values[log_prefix + "contrastive_loss"] = contrastive_loss
    log_values[log_prefix + "mean_geo_diff"] = mean_geo_error
    log_values[log_prefix + "mean_order"] = mean_order_error

    if l2_reg_weight > 0:
        mse_criterion = nn.MSELoss(reduction="sum")
        phi_norm = phi.norm(dim=1, keepdim=True)
        phi_rotated_norm = phi_rotated.norm(dim=1, keepdim=True)
        l2_reg = mse_criterion(phi_norm, torch.zeros_like(phi_norm)) + mse_criterion(
            phi_rotated_norm, torch.zeros_like(phi_rotated_norm)
        )

        contrastive_loss = contrastive_loss + l2_reg_weight * l2_reg
        log_values[log_prefix + "l2_reg_loss"] = l2_reg_weight * l2_reg

    return contrastive_loss, log_values


def compute_occlusion_infonce_loss(
    model: nn.Module,
    points: torch.Tensor,
    normalize_features: bool = True,
    weighting: str = "dist",
    l2_reg_weight: float = 0.0,
    center=None,
    log_prefix: str = "",
) -> Tuple[torch.Tensor, Dict]:
    """
    Computes the InfoNCE loss for a model given a batch of points.
    https://arxiv.org/pdf/2211.09325.pdf equation 19

    Args:
        model: the model to compute the loss for
        points: the batch of points to compute the loss for, [B, N, D]
        normalize_features: whether to normalize the features before computing the loss
        weighting: the weighting scheme to use for the loss
        l2_reg_weight: the weight of the l2 regularization term
        log_prefix: the prefix to use for logging

    Returns:
        contrastive_loss: the InfoNCE loss
        log_values: a dictionary of values to log
    """

    if center is None:
        points_centered_xyz = points[:, :, :3] - points[:, :, :3].mean(
            dim=1, keepdims=True
        )
        points_centered = torch.cat([points_centered_xyz, points[:, :, 3:]], dim=-1)
    else:
        points_centered_xyz = points[:, :, :3] - center
        points_centered = torch.cat([points_centered_xyz, points[:, :, 3:]], dim=-1)

    phi = model(points_centered.transpose(-1, -2))

    if normalize_features:
        phi = F.normalize(phi, dim=1)

    points_occ_list = []
    phi_occ_list = []
    mask_list = []
    for batch_idx in range(points.shape[0]):
        max_attempts = 3
        while max_attempts > 0:
            if max_attempts == 0:
                points_occ = points[batch_idx, :, :3]
                mask = torch.ones(points_occ.shape[0], dtype=torch.bool)
                break

            if np.random.rand() > 0.5:
                points_occ_xyz, mask = ball_occlusion(
                    points[batch_idx, :, :3], return_mask=True
                )
            else:
                points_occ_xyz, mask = plane_occlusion(
                    points[batch_idx, :, :3], return_mask=True
                )
            points_occ = torch.cat(
                [points_occ_xyz, points[batch_idx, mask, 3:]], dim=-1
            )

            if points_occ.shape[0] <= points.shape[1] // 2:
                max_attempts -= 1
                continue
            else:
                break

        phi_occ_single = model(points_occ.unsqueeze(0).transpose(-1, -2))

        if normalize_features:
            phi_occ_single = F.normalize(phi_occ_single, dim=1)

        points_occ_list.append(points_occ)
        mask_list.append(mask)
        phi_occ_list.append(phi_occ_single)

    if weighting.lower() == "mask":
        w = dist2weight(points_centered)
    elif weighting.lower() == "dist":
        w = dist2weight(points, func=lambda x: torch.tanh(10 * x))
    else:
        w = None

    contrastive_loss = 0
    mean_order_error = 0
    mean_geo_error = 0
    for batch_idx, (phi_occ, mask) in enumerate(zip(phi_occ_list, mask_list)):
        # mask is a 1024 length boolean tensor
        # Select the points that are not occluded
        phi_selected = phi[batch_idx, :, mask].unsqueeze(0)
        w_selected = w[batch_idx, mask, mask].unsqueeze(0)

        contrastive_loss, similarity = infonce_loss(
            phi_selected, phi_occ, weights=w_selected, temperature=0.1
        )

        mean_order_error = mean_order(similarity)
        mean_geo_error = mean_geo_diff(
            similarity, points[batch_idx, mask, :3].unsqueeze(0)
        )

        contrastive_loss += contrastive_loss
        mean_order_error += mean_order_error
        mean_geo_error += mean_geo_error

    contrastive_loss = contrastive_loss / points.shape[0]
    mean_order_error = mean_order_error / points.shape[0]
    mean_geo_error = mean_geo_error / points.shape[0]

    log_values = {}
    log_values[log_prefix + "occ_contrastive_loss"] = contrastive_loss
    log_values[log_prefix + "occ_mean_geo_diff"] = mean_geo_error
    log_values[log_prefix + "occ_mean_order"] = mean_order_error

    return contrastive_loss, log_values
