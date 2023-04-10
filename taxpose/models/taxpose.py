import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from taxpose.nets.transformer_flow_pm import BrianChuerAdapter
from third_party.dcp.model import DGCNN

mse_criterion = nn.MSELoss(reduction="sum")


def dense_flow_loss(points, flow_pred, trans_gt):
    flow_gt = trans_gt.transform_points(points) - points
    loss = mse_criterion(
        flow_pred,
        flow_gt,
    )
    return loss


class SE3LossTheirs(nn.Module):
    def forward(self, R_pred, R_gt, t_pred, t_gt):
        batch_size = len(t_gt)
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)

        R_loss = F.mse_loss(torch.matmul(R_pred.transpose(2, 1), R_gt), identity)
        t_loss = F.mse_loss(t_pred, t_gt)

        loss = R_loss + t_loss
        return loss, R_loss, t_loss


class BrianChuerLoss(nn.Module):
    def forward(self, action_pos, T_pred, T_gt, Fx):
        pred_flow_action = (T_gt.transform_points(action_pos) - action_pos).detach()

        loss = brian_chuer_loss(
            pred_T_action=T_pred,
            gt_T_action=T_gt,
            points_trans_action=action_pos,
            pred_flow_action=Fx,
            points_action=T_gt.transform_points(action_pos),
        )
        return loss


def brian_chuer_loss(
    pred_T_action,
    gt_T_action,
    points_trans_action,
    pred_flow_action,
    points_action,
    action_weight=1.0,
    smoothness_weight=0.1,
    consistency_weight=1.0,
):
    induced_flow_action = (
        pred_T_action.transform_points(points_trans_action) - points_trans_action
    ).detach()
    pred_points_action = pred_T_action.transform_points(
        points_trans_action
    )  # pred_points_action =  T0^-1@points_trans_action

    # pred_T_action=T0^-1
    # gt_T_action = T0.inverse()

    point_loss_action = mse_criterion(pred_points_action, points_action)

    point_loss = action_weight * point_loss_action

    dense_loss = dense_flow_loss(
        points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
    )

    # Loss associated flow vectors matching a consistent rigid transform
    smoothness_loss_action = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    )

    smoothness_loss = action_weight * smoothness_loss_action

    loss = (
        point_loss
        + smoothness_weight * smoothness_loss
        + consistency_weight * dense_loss
    )

    return loss


class DummyParams(nn.Module):
    """Dummy model that returns random (trainable) parameters for testing purposes."""

    def __init__(self, embedding_dim, act_attn):
        super().__init__()
        out_chan = 1 if act_attn else 0
        self.actor_params = nn.Parameter(
            torch.empty((200, embedding_dim + out_chan)), requires_grad=True
        )
        self.anchor_params = nn.Parameter(
            torch.empty((1800, embedding_dim)), requires_grad=True
        )
        init.uniform_(self.actor_params)
        init.uniform_(self.anchor_params)

    def forward(self, actor_data, anchor_data):
        return self.actor_params, self.anchor_params


class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.emb_dims = 16
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        if len(input) == 5 and input[4] is not None:
            aws = input[4].unsqueeze(1)
            # breakpoint()
        else:
            aws = 1.0
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(
            src_embedding.transpose(2, 1).contiguous(), tgt_embedding
        ) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(
            aws * src_centered, src_corr_centered.transpose(2, 1).contiguous()
        )

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            # print(H[i])
            r_pert = (torch.rand(3, 3) * 1e-7).to(H[i].device)
            r_pert = 0.0
            u, s, v = torch.linalg.svd(H[i] + r_pert)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.linalg.svd(H[i] + r_pert)
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(
            dim=2, keepdim=True
        )
        return R, t.view(batch_size, 3)


class TAXPoseModel(nn.Module):
    def __init__(self, arch="pn", attn=True, embedding_dim=512):
        super().__init__()
        self.arch = arch
        if arch == "dummy":
            self.net = DummyParams(16, False)
        elif arch == "pn":
            raise ValueError("PN is not supported yet")
            # self.action_net = PN2Dense(in_channels=0, out_channels=16)
            # self.anchor_net = PN2Dense(in_channels=0, out_channels=16)
        elif arch == "dgcnn":
            chan = 1 if attn else 0
            self.action_net = DGCNN(embedding_dim + chan)
            self.anchor_net = DGCNN(embedding_dim)
        elif arch == "brianchuer" or arch == "brianchuer-loss":
            self.net = BrianChuerAdapter(emb_dims=embedding_dim)
        elif arch == "brianchuer-gc":
            self.net = BrianChuerAdapter(emb_dims=embedding_dim, gc=True)

        self.attn = attn

        self.head = SVDHead()

    @staticmethod
    def norm_scale(pos):
        mean = pos.mean(dim=1).unsqueeze(1)
        pos = pos - mean
        scale = pos.abs().max(dim=2)[0].max(dim=1)[0]
        pos = pos / (scale.view(-1, 1, 1) + 1e-8)
        return pos

    def forward(self, action, anchor):
        X, Y = action.pos, anchor.pos
        aws = None
        if self.arch == "dummy":
            Ex, Ey = self.net(action, anchor)
        # elif self.arch == "pn":
        #     Ex = self.action_net(norm_scale_batch(action))
        #     Ey = self.anchor_net(norm_scale_batch(anchor))
        elif self.arch == "dgcnn":
            Xns = self.norm_scale(X.view(action.num_graphs, -1, 3)).transpose(-1, -2)
            Yns = self.norm_scale(Y.view(anchor.num_graphs, -1, 3)).transpose(-1, -2)
            Ex = self.action_net(Xns)
            Ey = self.anchor_net(Yns)
            if self.attn:
                aws = torch.softmax(Ex[:, -1, :], dim=-1) * Ex.shape[-1]
                Ex = Ex[:, :-1, :]
            # breakpoint()
        elif self.arch == "brianchuer" or self.arch == "brianchuer-loss":
            Xs = X.view(action.num_graphs, -1, 3)
            Ys = Y.view(anchor.num_graphs, -1, 3)
            R_pred, t_pred, pred_T_action, Fx, Fy = self.net(Xs, Ys)
            return R_pred, t_pred, None, pred_T_action, Fx, Fy
        elif self.arch == "brianchuer-gc":
            Xs = X.view(action.num_graphs, -1, 3)
            Ys = Y.view(anchor.num_graphs, -1, 3)
            R_pred, t_pred, pred_T_action, Fx, Fy = self.net(
                Xs, Ys, action.loc.unsqueeze(-1)
            )
            return R_pred, t_pred, None, pred_T_action, Fx, Fy
        else:
            raise ValueError("bad")

        if self.arch in {"dummy", "pn"}:
            Ex = Ex.unsqueeze(0).transpose(-1, -2)
            Ey = Ey.unsqueeze(0).transpose(-1, -2)
            X = X.unsqueeze(0).transpose(-1, -2)
            Y = Y.unsqueeze(0).transpose(-1, -2)
        else:
            X = X.view(action.num_graphs, -1, 3).transpose(-1, -2)
            Y = Y.view(anchor.num_graphs, -1, 3).transpose(-1, -2)

        R_pred, t_pred = self.head(Ex, Ey, X, Y, aws)

        return R_pred, t_pred, aws
