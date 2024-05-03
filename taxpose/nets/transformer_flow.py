#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.pointnet import PointNet
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.vn_dgcnn import VN_DGCNN, VNArgs
from third_party.dcp.model import DGCNN


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, encoder_cfg):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_nn = create_embedding_network(encoder_cfg)

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)

        points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points

        return points_embedding


class CorrespondenceFlow_DiffEmbMLP(nn.Module):
    def __init__(self, encoder_cfg, cycle=True, center_feature=True):
        super(CorrespondenceFlow_DiffEmbMLP, self).__init__()
        self.cycle = cycle

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        emb_dims = emb_nn.encoder

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

        outputs = {
            "flow_action": flow_action,
        }

        if self.cycle:
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=None,
            ).permute(0, 2, 1)
            outputs["flow_anchor"] = flow_anchor

        return outputs


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

    def forward(self, *input, scores=None, return_embedding=False):
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


def create_embedding_network(cfg) -> nn.Module:
    if cfg.name == "dgcnn":
        network: nn.Module = DGCNN(emb_dims=cfg.emb_dims)
    elif cfg.name == "vn_dgcnn":
        args = VNArgs()
        network = VN_DGCNN(args, num_part=cfg.emb_dims, gc=False)
    else:
        raise ValueError(f"Unknown embedding network type: {cfg.name}")

    return network


class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        cycle=True,
        center_feature=False,
        pred_weight=True,
        residual_on=True,
        freeze_embnn=False,
        return_attn=True,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.cycle = cycle

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        emb_dims = encoder_cfg.emb_dims

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

        return outputs


class ModelConfig(Protocol):
    model_type: str


@dataclass
class ResidualFlowDiffEmbTransformerConfig:
    model_type: ClassVar[str] = "residual_flow_diff_emb_transformer"

    encoder: Any

    cycle: bool
    center_feature: bool
    pred_weight: bool
    residual_on: bool
    freeze_embnn: bool
    return_attn: bool


@dataclass
class CorrespondenceFlowDiffEmbMLPConfig:
    model_type: ClassVar[str] = "correspondence_flow_diff_emb_mlp"

    encoder: Any

    cycle: bool
    center_feature: bool


def create_network(cfg: ModelConfig) -> nn.Module:
    # Create the network
    if cfg.model_type == "residual_flow_diff_emb_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network: nn.Module = ResidualFlow_DiffEmbTransformer(
            encoder_cfg=r_cfg.encoder,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            pred_weight=r_cfg.pred_weight,
            residual_on=r_cfg.residual_on,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
        )
    elif cfg.model_type == "correspondence_flow_diff_emb_mlp":
        c_cfg = cast(CorrespondenceFlowDiffEmbMLPConfig, cfg)
        network = CorrespondenceFlow_DiffEmbMLP(
            encoder_cfg=c_cfg.encoder,
            cycle=c_cfg.cycle,
            center_feature=c_cfg.center_feature,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    return network
