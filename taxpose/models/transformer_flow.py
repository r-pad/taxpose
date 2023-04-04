#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.pointnet import PointNet
from third_party.dcp.model import (
    DGCNN,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
)


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn="dgcnn"):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)

        points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points

        return points_embedding


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        n_blocks=1,
        dropout=0.0,
        ff_dims=1024,
        n_heads=4,
        return_attn=False,
        bidirectional=True,
    ):
        super(BidirectionalTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
            Decoder(
                DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout),
                self.N,
            ),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
        )

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        src_attn = self.model.decoder.layers[-1].src_attn.attn

        if self.bidirectional:
            tgt_embedding = (
                self.model(src, tgt, None, None).transpose(2, 1).contiguous()
            )
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn

            if self.return_attn:
                return src_embedding, tgt_embedding, src_attn, tgt_attn
            return src_embedding, tgt_embedding

        if self.return_attn:
            return src_embedding, src_attn
        return src_embedding


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
        if return_flow_component:
            return {
                "full_flow": corr_flow_weight,
                "residual_flow": residual_flow,
                "corr_flow": corr_flow,
                "corr_points": corr_points,
                "scores": scores,
            }
        return corr_flow_weight


class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        inital_sampling_ratio=0.2,
        pred_weight=True,
        residual_on=True,
        freeze_embnn=False,
        use_transformer_attention=True,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.use_transformer_attention = use_transformer_attention

        self.transformer_action = BidirectionalTransformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = BidirectionalTransformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
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
        if self.freeze_embnn:
            action_embedding = self.emb_nn_action(action_points_dmean).detach()
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean).detach()
        else:
            action_embedding = self.emb_nn_action(action_points_dmean)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        action_embedding_tf, action_attn = self.transformer_action(
            action_embedding, anchor_embedding
        )
        anchor_embedding_tf, anchor_attn = self.transformer_anchor(
            anchor_embedding, action_embedding
        )

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        if not self.use_transformer_attention:
            action_attn = None
            anchor_attn = None

        action_attn = action_attn.mean(dim=1)
        if self.return_flow_component:
            flow_output_action = self.head_action(
                action_embedding_tf,
                anchor_embedding_tf,
                action_points,
                anchor_points,
                scores=action_attn,
                return_flow_component=self.return_flow_component,
            )
            flow_action = flow_output_action["full_flow"].permute(0, 2, 1)
            residual_flow_action = flow_output_action["residual_flow"].permute(0, 2, 1)
            corr_flow_action = flow_output_action["corr_flow"].permute(0, 2, 1)
        else:
            flow_action = self.head_action(
                action_embedding_tf,
                anchor_embedding_tf,
                action_points,
                anchor_points,
                scores=action_attn,
                return_flow_component=self.return_flow_component,
            ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            if self.return_flow_component:
                flow_output_anchor = self.head_anchor(
                    anchor_embedding_tf,
                    action_embedding_tf,
                    anchor_points,
                    action_points,
                    scores=anchor_attn,
                    return_flow_component=self.return_flow_component,
                )
                flow_anchor = flow_output_anchor["full_flow"].permute(0, 2, 1)
                residual_flow_anchor = flow_output_anchor["residual_flow"].permute(
                    0, 2, 1
                )
                corr_flow_anchor = flow_output_anchor["corr_flow"].permute(0, 2, 1)
            else:
                flow_anchor = self.head_anchor(
                    anchor_embedding_tf,
                    action_embedding_tf,
                    anchor_points,
                    action_points,
                    scores=anchor_attn,
                    return_flow_component=self.return_flow_component,
                ).permute(0, 2, 1)
            if self.return_flow_component:
                return {
                    "flow_action": flow_action,
                    "flow_anchor": flow_anchor,
                    "residual_flow_action": residual_flow_action,
                    "residual_flow_anchor": residual_flow_anchor,
                    "corr_flow_action": corr_flow_action,
                    "corr_flow_anchor": corr_flow_anchor,
                    "action_attn": action_attn,
                    "anchor_attn": anchor_attn,
                    "corr_points_action": flow_output_action["corr_points"],
                    "scores_action": flow_output_action["scores"],
                    "corr_points_anchor": flow_output_anchor["corr_points"],
                    "scores_anchor": flow_output_anchor["scores"],
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
                "action_attn": action_attn,
                "corr_points_action": flow_output_action["corr_points"],
                "scores_action": flow_output_action["scores"],
            }
        else:
            return flow_action
