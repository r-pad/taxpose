#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import math

import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.pointnet import PointNet
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.tv_mlp import MLP as TVMLP
from taxpose.nets.vn_dgcnn import VN_DGCNN, VNArgs
from taxpose.utils.multilateration import estimate_p
from third_party.dcp.model import DGCNN


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn="dgcnn"):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn_dgcnn":
            self.emb_nn = VN_DGCNN(VNArgs(), num_part=self.emb_dims, gc=False)
        else:
            raise Exception("Not implemented")

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)

        points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points

        return points_embedding


class CorrespondenceFlow_DiffEmbMLP(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn", center_feature=True):
        super(CorrespondenceFlow_DiffEmbMLP, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle

        if emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.center_feature = center_feature

        self.transformer_action = MLP(emb_dims=emb_dims)
        self.transformer_anchor = MLP(emb_dims=emb_dims)
        self.head_action = CorrespondenceMLPHead(emb_dims=emb_dims)
        self.head_anchor = CorrespondenceMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        action_embedding = self.emb_nn_action(action_points_dmean)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        action_embedding_tf = self.transformer_action(action_embedding)
        # action_embedding_tf: Batch, emb_dim, num_points
        # action_attn: Batch, 4, num_points, num_points
        anchor_embedding_tf = self.transformer_anchor(anchor_embedding)
        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=None,
        ).permute(0, 2, 1)

        if self.cycle:
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=None,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor
        return flow_action


class CorrespondenceMLPHead(nn.Module):
    """
    Output correspondence flow and weight
    """

    def __init__(self, emb_dims=512):
        super(CorrespondenceMLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, 64, 64, 64, 128, 512]),
            # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(512, 1, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            if len(input) <= 4:
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points
        weight = self.proj_flow(action_embedding)
        corr_flow_weight = torch.concat([corr_flow, weight], dim=1)

        return corr_flow_weight


class MLP(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(emb_dims, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, emb_dims)

    def forward(self, x):
        # x = [batch size, emb_dims, num_points]
        batch_size, _, num_points = x.shape
        x = x.permute(0, -1, -2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        h_1 = F.relu(self.input_fc(x))
        # batch size*num_points, 100
        h_2 = F.relu(self.hidden_fc(h_1))

        # batch size*num_points, output dim
        y_pred = self.output_fc(h_2)
        # batch size, num_points, output dim
        y_pred = y_pred.view(batch_size, num_points, -1)
        # batch size, emb_dims, num_points
        y_pred = y_pred.permute(0, 2, 1)

        return y_pred


class MLPHead(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8]),
            nn.Conv1d(emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow


class MLPHeadWeight(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHeadWeight, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8]),
            nn.Conv1d(emb_dims // 8, 4, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow


class ResidualMLPHead(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512, pred_weight=True, residual_on=True):
        super(ResidualMLPHead, self).__init__()

        self.emb_dims = emb_dims
        if self.emb_dims < 10:
            self.proj_flow = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )
        else:
            self.proj_flow = nn.Sequential(
                PointNet([emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8]),
                nn.Conv1d(emb_dims // 8, 3, kernel_size=1, bias=False),
            )
        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

        self.residual_on = residual_on

    def forward(
        self, *input, scores=None, return_flow_component=False, return_embedding=False
    ):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]

        if scores is None:
            if len(input) <= 4:
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)
        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        if self.residual_on:
            flow = residual_flow + corr_flow
        else:
            flow = corr_flow

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            "full_flow": corr_flow_weight,
            "residual_flow": residual_flow,
            "corr_flow": corr_flow,
            "corr_points": corr_points,
            "scores": scores,
        }


class MLPKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = TVMLP(2 * feature_dim, [300, 100, 1])

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
    ):
        super().__init__()

        self.emb_dims = emb_dims
        self.n_kps = n_kps
        self.last_attn = last_attn

        self.kernel = MLPKernel(self.emb_dims - int(last_attn))
        # self.kernel = NormKernel(self.emb_dims)
        self.sample = sample

        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims - int(last_attn), 64, 64, 64, 128, 512]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

    def forward(
        self, *input, scores=None, return_flow_component=False, return_embedding=False
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


class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        pred_weight=True,
        residual_on=True,
        freeze_embnn=False,
        return_attn=True,
        multilaterate=False,
        sample: bool = False,
        mlat_nkps: int = 100,
        break_symmetry=False,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        self.break_symmetry = break_symmetry
        if emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn_dgcnn":
            args = VNArgs()
            self.emb_nn_action = VN_DGCNN(args, num_part=self.emb_dims, gc=False)
            self.emb_nn_anchor = VN_DGCNN(args, num_part=self.emb_dims, gc=False)
        else:
            raise Exception("Not implemented")
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.return_attn = return_attn

        self.transformer_action = CustomTransformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        self.transformer_anchor = CustomTransformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        if multilaterate:
            self.head_action = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
            )
            self.head_anchor = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
            )
        else:
            self.head_action = ResidualMLPHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                residual_on=self.residual_on,
            )
            self.head_anchor = ResidualMLPHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                residual_on=self.residual_on,
            )

        if self.break_symmetry:
            # We're basically putting a few MLP layers in on top of the invariant module.
            emb_dims_sym = self.emb_dims + 1
            self.proj_flow_symmetry_labels_action = nn.Sequential(
                PointNet([emb_dims_sym, emb_dims_sym * 2, emb_dims_sym * 4]),
                nn.Conv1d(emb_dims_sym * 4, self.emb_dims, kernel_size=1, bias=False),
            )
            self.proj_flow_symmetry_labels_anchor = nn.Sequential(
                PointNet([emb_dims_sym, emb_dims_sym * 2, emb_dims_sym * 4]),
                nn.Conv1d(emb_dims_sym * 4, self.emb_dims, kernel_size=1, bias=False),
            )

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)

        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points

        action_embedding = self.emb_nn_action(action_points_dmean)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        if self.freeze_embnn:
            action_embedding = action_embedding.detach()
            anchor_embedding = anchor_embedding.detach()

        if self.break_symmetry:
            # Add a symmetry label to the embeddings.
            action_sym_cls = input[2].permute(0, 2, 1)
            anchor_sym_cls = input[3].permute(0, 2, 1)

            action_embedding_stack = torch.cat(
                [action_embedding, action_sym_cls], axis=1
            )
            anchor_embedding_stack = torch.cat(
                [anchor_embedding, anchor_sym_cls], axis=1
            )

            action_embedding = self.proj_flow_symmetry_labels_action(
                action_embedding_stack
            )

            anchor_embedding = self.proj_flow_symmetry_labels_anchor(
                anchor_embedding_stack
            )

        # tilde_phi, phi are both B,512,N
        # Get the new cross-attention embeddings.
        transformer_action_outputs = self.transformer_action(
            action_embedding, anchor_embedding
        )
        transformer_anchor_outputs = self.transformer_anchor(
            anchor_embedding, action_embedding
        )
        action_embedding_tf = transformer_action_outputs["src_embedding"]
        action_attn = transformer_action_outputs["src_attn"]
        anchor_embedding_tf = transformer_anchor_outputs["src_embedding"]
        anchor_attn = transformer_anchor_outputs["src_attn"]

        if not self.return_attn:
            action_attn = None
            anchor_attn = None

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        if action_attn is not None:
            action_attn = action_attn.mean(dim=1)

        head_action_output = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        )
        flow_action = head_action_output["full_flow"].permute(0, 2, 1)
        residual_flow_action = head_action_output["residual_flow"].permute(0, 2, 1)
        corr_flow_action = head_action_output["corr_flow"].permute(0, 2, 1)
        corr_points_action = head_action_output["corr_points"].permute(0, 2, 1)

        outputs = {
            "flow_action": flow_action,
            "residual_flow_action": residual_flow_action,
            "corr_flow_action": corr_flow_action,
            "corr_points_action": corr_points_action,
        }

        if "P_A" in head_action_output:
            original_points_action = head_action_output["P_A"].permute(0, 2, 1)
            outputs["original_points_action"] = original_points_action
            outputs["sampled_ixs_action"] = head_action_output["A_ixs"]

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            head_anchor_output = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            )
            flow_anchor = head_anchor_output["full_flow"].permute(0, 2, 1)
            residual_flow_anchor = head_anchor_output["residual_flow"].permute(0, 2, 1)
            corr_flow_anchor = head_anchor_output["corr_flow"].permute(0, 2, 1)
            corr_points_anchor = head_anchor_output["corr_points"].permute(0, 2, 1)

            outputs = {
                **outputs,
                "flow_anchor": flow_anchor,
                "residual_flow_anchor": residual_flow_anchor,
                "corr_flow_anchor": corr_flow_anchor,
                "corr_points_anchor": corr_points_anchor,
            }

            if "P_A" in head_anchor_output:
                original_points_anchor = head_anchor_output["P_A"].permute(0, 2, 1)
                outputs["original_points_anchor"] = original_points_anchor
                outputs["sampled_ixs_anchor"] = head_anchor_output["A_ixs"]

        return outputs
