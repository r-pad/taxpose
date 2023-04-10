import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import (
    Rotate,
    Transform3d,
    Translate,
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    so3_rotation_angle,
)
from torch.nn import functional as F

mse_criterion = nn.MSELoss(reduction="sum")


def to_transform3d(x, rot_function=rotation_6d_to_matrix):
    trans = x[:, :3]
    rot = x[:, 3:]
    return (
        Transform3d(device=x.device)
        .compose(Rotate(rot_function(rot), device=rot.device))
        .translate(trans)
    )


def transform_points(points, x, rot_function=rotation_6d_to_matrix):
    t = x[:, :3]
    R = rot_function(x[:, 3:])
    return (torch.bmm(R, points.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)


def transform3d_to(T, device):
    T = T.to(device)
    T = T.to(device)
    T._transforms = [t.to(device) for t in T._transforms]
    return T


def random_se3(
    N, rot_var=np.pi / 180 * 5, trans_var=0.1, device=None, fix_random=False
):
    axis_angle_random = torch.randn(N, 3, device=device)
    rot_ratio = (
        torch.rand(1).item()
        * rot_var
        / torch.norm(axis_angle_random, dim=1).max().item()
    )
    constrained_axix_angle = rot_ratio * axis_angle_random  # max angle is rot_var
    R = axis_angle_to_matrix(constrained_axix_angle)
    random_translation = torch.randn(N, 3, device=device)
    translation_ratio = trans_var / torch.norm(random_translation, dim=1).max().item()
    t = torch.rand(1).item() * translation_ratio * random_translation
    return Rotate(R, device=device).translate(t)


def get_degree_angle(T):
    angle_rad_T = (
        so3_rotation_angle(T.get_matrix()[:, :3, :3], eps=1e-2) * 180 / np.pi
    )  # B

    max = torch.max(angle_rad_T).item()
    min = torch.min(angle_rad_T).item()
    mean = torch.mean(angle_rad_T).item()
    return max, min, mean


def get_translation(T):
    t = T.get_matrix()[:, 3, :3]  # B,3
    t_norm = torch.norm(t, dim=1)  # B
    max = torch.max(t_norm).item()
    min = torch.min(t_norm).item()
    mean = torch.mean(t_norm).item()
    return max, min, mean


def rotation_se3(N, axis, angle_degree, device=None):
    """
    Args
        axis: torch tensor of shape (3)
        angle_degree: int
    """
    angle_rad = angle_degree * (np.pi / 180)
    axis_angle = angle_rad * axis
    axis_angle = axis_angle.unsqueeze(0)  # (1,3)
    axis_angle = torch.repeat_interleave(axis_angle, N, dim=0)  # (N,3)
    R = axis_angle_to_matrix(axis_angle.to(device))
    t = torch.zeros(N, 3, device=device)
    return Rotate(R, device=device).translate(t)


def pure_translation_se3(N, t, device=None):
    """
    Args
        t: torch tensor of shape (3)
    """
    axis = torch.tensor([0, 0, 1])
    axis_angle = 0.0 * axis
    axis_angle = axis_angle.unsqueeze(0)  # (1,3)
    axis_angle = torch.repeat_interleave(axis_angle, N, dim=0)  # (N,3)
    R = axis_angle_to_matrix(axis_angle.to(device))  # identity
    assert torch.allclose(
        torch.eye(3).to(device), R[0]
    ), "R should be identity for pure translation se3"
    t = torch.repeat_interleave(t.unsqueeze(0), N, dim=0).to(device)  # N,3
    return Rotate(R, device=device).translate(t)


def symmetric_orthogonalization(M):
    """Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)
    M: should have size [batch_size, 3, 3]
    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    return R


def flow2pose(
    xyz,
    flow,
    weights=None,
    return_transform3d=False,
    normalization_scehme="l1",
    x=None,
    temperature=1,
):
    """
    @param xyz: (batch, num_points, 3)
    @param flow: (batch, num_points,3)
    @param weights: (batch, num_points)
    @param normalization_scehme: {'l1, 'softmax'}
    @param x: flow prediction
    """
    assert normalization_scehme in [
        "l1",
        "softmax",
    ], "normalization_scehme: {} is not currently supported!".format(
        normalization_scehme
    )
    if weights is None:
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    if normalization_scehme == "l1":
        w = F.normalize(weights, p=1.0, dim=-1).unsqueeze(-1)  # B, num_points, 1
    elif normalization_scehme == "softmax":
        softmax_operator = torch.nn.Softmax(dim=-1)
        # B, num_points, 1
        w = softmax_operator(weights / temperature).unsqueeze(-1)
    # if not torch.allclose(w.sum(1), torch.ones(w.sum(1).shape).cuda()):
    #     import pdb
    #     pdb.set_trace()
    assert torch.allclose(
        w.sum(1), torch.ones(w.sum(1).shape).cuda()
    ), "flow weights does not sum to 1 for each batch element"
    xyz_mean = (w * xyz).sum(dim=1, keepdims=True)
    xyz_demean = xyz - xyz_mean

    flow_mean = (w * flow).sum(dim=1, keepdims=True)
    # xyz_trans = xyz_demean + flow - flow_mean
    xyz_trans = ((xyz + flow) * w).sum(dim=1, keepdims=True)

    X = torch.bmm(xyz_demean.transpose(-2, -1), w * xyz_trans)

    R = symmetric_orthogonalization(X)
    t = (flow_mean + xyz_mean - torch.bmm(xyz_mean, R)).squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    return R, t


eps = 1e-9


def dualflow2pose(
    xyz_src,
    xyz_tgt,
    flow_src,
    flow_tgt,
    weights_src=None,
    weights_tgt=None,
    return_transform3d=False,
    normalization_scehme="l1",
    temperature=1,
):
    assert normalization_scehme in [
        "l1",
        "softmax",
    ], "normalization_scehme: {} is not currently supported!".format(
        normalization_scehme
    )
    if weights_src is None:
        weights_src = torch.ones(xyz_src.shape[:-1], device=xyz_src.device)

    if weights_tgt is None:
        weights_tgt = torch.ones(xyz_tgt.shape[:-1], device=xyz_tgt.device)

    if normalization_scehme == "l1":
        w_src = F.normalize(weights_src, p=1.0, dim=-1).unsqueeze(-1)
        w_tgt = F.normalize(weights_tgt, p=1.0, dim=-1).unsqueeze(-1)
    elif normalization_scehme == "softmax":
        softmax_operator = torch.nn.Softmax(dim=-1)
        w_src = softmax_operator(weights_src / temperature).unsqueeze(-1)
        w_tgt = softmax_operator(weights_tgt / temperature).unsqueeze(-1)
    assert torch.allclose(
        w_src.sum(1), torch.ones(w_src.sum(1).shape).cuda()
    ), "flow src weights does not sum to 1 for each batch element"
    assert torch.allclose(
        w_tgt.sum(1), torch.ones(w_tgt.sum(1).shape).cuda()
    ), "flow tgt weights does not sum to 1 for each batch element"

    xyz_mean_src = (w_src * xyz_src).sum(dim=1, keepdims=True)

    xyz_centered_src = xyz_src - xyz_mean_src

    xyz_mean_tgt = (w_tgt * xyz_tgt).sum(dim=1, keepdims=True)
    xyz_centered_tgt = xyz_tgt - xyz_mean_tgt

    flow_mean_src = (w_src * flow_src).sum(dim=1, keepdims=True)
    flow_centered_src = flow_src - flow_mean_src
    flow_mean_tgt = (w_tgt * flow_tgt).sum(dim=1, keepdims=True)
    flow_centered_tgt = flow_tgt - flow_mean_tgt

    w = torch.cat([w_src, w_tgt], dim=1)
    xyz_1 = torch.cat([xyz_centered_src, xyz_centered_tgt + flow_centered_tgt], dim=1)
    xyz_2 = torch.cat([xyz_centered_src + flow_centered_src, xyz_centered_tgt], dim=1)

    X = torch.bmm(xyz_1.transpose(-2, -1), w * xyz_2)

    R = symmetric_orthogonalization(X)
    t_src = flow_mean_src + xyz_mean_src - torch.bmm(xyz_mean_src, R)
    t_tgt = xyz_mean_tgt - torch.bmm(flow_mean_tgt + xyz_mean_tgt, R)

    t = (
        (w_src.shape[1] * t_src + w_tgt.shape[1] * t_tgt)
        / (w_src.shape[1] + w_tgt.shape[1])
    ).squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    return R, t


def dualflow2translation(
    xyz_src,
    xyz_tgt,
    flow_src,
    flow_tgt,
    weights_src=None,
    weights_tgt=None,
    return_transform3d=False,
    normalization_scehme="l1",
    temperature=1,
):
    assert normalization_scehme in [
        "l1",
        "softmax",
    ], "normalization_scehme: {} is not currently supported!".format(
        normalization_scehme
    )
    if weights_src is None:
        weights_src = torch.ones(xyz_src.shape[:-1], device=xyz_src.device)

    if weights_tgt is None:
        weights_tgt = torch.ones(xyz_tgt.shape[:-1], device=xyz_tgt.device)

    if normalization_scehme == "l1":
        w_src = F.normalize(weights_src, p=1.0, dim=-1).unsqueeze(-1)
        w_tgt = F.normalize(weights_tgt, p=1.0, dim=-1).unsqueeze(-1)
    elif normalization_scehme == "softmax":
        softmax_operator = torch.nn.Softmax(dim=-1)
        w_src = softmax_operator(weights_src / temperature).unsqueeze(-1)
        w_tgt = softmax_operator(weights_tgt / temperature).unsqueeze(-1)
    assert torch.allclose(
        w_src.sum(1), torch.ones(w_src.sum(1).shape).cuda()
    ), "flow src weights does not sum to 1 for each batch element"
    assert torch.allclose(
        w_tgt.sum(1), torch.ones(w_tgt.sum(1).shape).cuda()
    ), "flow tgt weights does not sum to 1 for each batch element"

    xyz_mean_src = (w_src * xyz_src).sum(dim=1, keepdims=True)
    xyz_mean_tgt = (w_tgt * xyz_tgt).sum(dim=1, keepdims=True)

    flow_mean_src = (w_src * flow_src).sum(dim=1, keepdims=True)
    flow_mean_tgt = (w_tgt * flow_tgt).sum(dim=1, keepdims=True)

    t_src = flow_mean_src + xyz_mean_src - xyz_mean_src
    t_tgt = xyz_mean_tgt - flow_mean_tgt + xyz_mean_tgt

    t = (
        (w_src.shape[1] * t_src + w_tgt.shape[1] * t_tgt)
        / (w_src.shape[1] + w_tgt.shape[1])
    ).squeeze(1)

    if return_transform3d:
        return Translate(t)
    return t


def dualflow2pose_joint(
    xyz,
    flow,
    polarity,
    weights=None,
    return_transform3d=False,
    normalization_scehme="l1",
):
    assert normalization_scehme in [
        "l1",
        "softmax",
    ], "normalization_scehme: {} is not currently supported!".format(
        normalization_scehme
    )
    if weights is None:
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    if normalization_scehme == "l1":
        w = F.normalize(weights, p=1.0, dim=-1).unsqueeze(-1)
    elif normalization_scehme == "softmax":
        softmax_operator = torch.nn.Softmax(dim=-1)
        w = softmax_operator(weights).unsqueeze(-1)
    assert torch.allclose(
        w.sum(1), torch.ones(w.sum(1).shape).cuda()
    ), "flow weights does not sum to 1 for each batch element"

    w_p = (polarity * weights).unsqueeze(-1)
    w_p_sum = w_p.sum(dim=1, keepdims=True)
    w_p = w_p / w_p_sum.clamp(min=eps)
    w_n = ((1 - polarity) * weights).unsqueeze(-1)
    w_n_sum = w_n.sum(dim=1, keepdims=True)
    w_n = w_n / w_n_sum.clamp(min=eps)

    xyz_mean_p = (w_p * xyz).sum(dim=1, keepdims=True)
    xyz_demean_p = xyz - xyz_mean_p
    xyz_mean_n = (w_n * xyz).sum(dim=1, keepdims=True)
    xyz_demean_n = xyz - xyz_mean_n

    flow_mean_p = (w_p * flow).sum(dim=1, keepdims=True)
    flow_demean_p = flow - flow_mean_p
    flow_mean_n = (w_n * flow).sum(dim=1, keepdims=True)
    flow_demean_n = flow - flow_mean_n

    mask = polarity.unsqueeze(-1).expand(-1, -1, 3) == 1
    xyz_1 = torch.where(mask, xyz_demean_p, xyz_demean_n + flow_demean_n)
    xyz_2 = torch.where(mask, xyz_demean_p + flow_demean_p, xyz_demean_n)

    X = torch.bmm(xyz_1.transpose(-2, -1), w * xyz_2)

    R = symmetric_orthogonalization(X)
    t_p = flow_mean_p + xyz_mean_p - torch.bmm(xyz_mean_p, R)
    t_n = xyz_mean_n - torch.bmm(flow_mean_n + xyz_mean_n, R)

    t = ((w_p_sum * t_p + w_n_sum * t_n) / (w_p_sum + w_n_sum)).squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    return R, t


def points2pose(
    xyz1, xyz2, weights=None, return_transform3d=False, normalization_scehme="l1"
):
    assert normalization_scehme in [
        "l1",
        "softmax",
    ], "normalization_scehme: {} is not currently supported!".format(
        normalization_scehme
    )
    if weights is None:
        weights = torch.ones(xyz1.shape[:-1], device=xyz1.device)
    if normalization_scehme == "l1":
        w = F.normalize(weights, p=1.0, dim=-1).unsqueeze(-1)
    elif normalization_scehme == "softmax":
        softmax_operator = torch.nn.Softmax(dim=-1)
        w = softmax_operator(weights).unsqueeze(-1)
    assert torch.allclose(
        w.sum(1), torch.ones(w.sum(1).shape).cuda()
    ), "flow weights does not sum to 1 for each batch element"
    xyz1_mean = (w * xyz1).sum(dim=1, keepdims=True)
    xyz1_demean = xyz1 - xyz1_mean

    xyz2_mean = (w * xyz2).sum(dim=1, keepdims=True)
    xyz2_demean = xyz2 - xyz2_mean

    X = torch.bmm(xyz1_demean.transpose(-2, -1), w * xyz2_demean)

    R = symmetric_orthogonalization(X)
    t = (xyz2_mean - torch.bmm(xyz1_mean, R)).squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    return R, t


def dense_flow_loss(points, flow_pred, trans_gt):
    flow_gt = trans_gt.transform_points(points) - points
    loss = mse_criterion(
        flow_pred,
        flow_gt,
    )
    return loss


def svd_flow_loss(points, flow_pred, points_tgt, weights_pred=None):
    T_pred = flow2pose(points, flow_pred, weights_pred, return_transform3d=True)
    points_pred = T_pred.transform_points(points)
    induced_flow = (points_pred - points).detach()

    point_loss = mse_criterion(
        points_pred,
        points_tgt,
    )

    consistency_loss = mse_criterion(
        flow_pred,
        induced_flow,
    )

    return point_loss, consistency_loss


def consistency_flow_loss(points, flow_pred, weights_pred=None):
    T_pred = flow2pose(
        points, flow_pred, weights_pred, return_transform3d=True
    ).detach()
    points_pred = T_pred.transform_points(points)
    induced_flow = points_pred - points

    consistency_loss = mse_criterion(
        flow_pred,
        induced_flow,
    )

    return consistency_loss


def consistency_dualflow_loss(
    points_trans_action,
    points_trans_anchor,
    pred_flow_action,
    pred_flow_anchor,
    pred_w_action,
    pred_w_anchor,
    weight_normalize="softmax",
):
    pred_T_action = dualflow2pose(
        xyz_src=points_trans_action,
        xyz_tgt=points_trans_anchor,
        flow_src=pred_flow_action,
        flow_tgt=pred_flow_anchor,
        weights_src=pred_w_action,
        weights_tgt=pred_w_anchor,
        return_transform3d=True,
        normalization_scehme=weight_normalize,
    )
    points_pred = pred_T_action.transform_points(points_trans_action)
    induced_flow_action = points_pred - points_trans_action

    consistency_loss = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    )

    return consistency_loss
