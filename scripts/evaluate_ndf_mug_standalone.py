import copy
import math
import os
import os.path as osp
import random
import signal
import time

import hydra
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import numpy as np
import pybullet as p
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from airobot import Robot, log_info, log_warn, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.share.globals import (
    bad_shapenet_bottles_ids_list,
    bad_shapenet_bowls_ids_list,
    bad_shapenet_mug_ids_list,
)
from ndf_robot.utils import path_util, util
from ndf_robot.utils.eval_gen_utils import (
    constraint_grasp_close,
    constraint_grasp_open,
    constraint_obj_world,
    get_ee_offset,
    object_is_still_grasped,
    process_demo_data_rack,
    process_demo_data_shelf,
    process_xq_data,
    process_xq_rs_data,
    safeCollisionFilterPair,
    safeRemoveConstraint,
    soft_grasp_close,
)
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.util import np2img
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import (
    Rotate,
    Transform3d,
    Translate,
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    so3_rotation_angle,
)
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor

# Commented out... so I don't have to bring them in.
# from equivariant_pose_graph.models.pointnet2.pointnet2_sem_seg import \
#     get_model as pointnet2
# from equivariant_pose_graph.models.pointnet2_geo import (PN2DenseParams,
#                                                          Pointnet2Dense)
# from equivariant_pose_graph.models.vnn.vn_layers import VNLinearLeakyReLU
# from equivariant_pose_graph.utils.transformer_utils import PositionalEncoding3D

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


class VN_PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(VN_PointNet, self).__init__()

        convs = []
        for j in range(len(layer_dims) - 1):
            convs.append(
                VNLinearLeakyReLU(
                    layer_dims[j],
                    layer_dims[j + 1],
                    dim=4,
                    negative_slope=0.0,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        return x


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn="dgcnn", inital_sampling_ratio=0.2):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "pointnet2ben":
            params = PN2DenseParams()
            params.sa1.ratio = inital_sampling_ratio
            self.emb_nn = Pointnet2Dense(emb_dims, params=params)
        elif emb_nn == "pointnet2ori":
            self.emb_nn = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn = VN_PointNet()

        else:
            raise Exception("Not implemented")

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)
        if self.emb_nn_name == "pointnet2ori":
            points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points
        else:
            points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points

        return points_embedding


class ReloadEquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn="dgcnn"):
        super(ReloadEquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "pointnet2ben":
            self.emb_nn = Pointnet2Dense(emb_dims)
        elif emb_nn == "pointnet2ori":
            self.emb_nn = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn = VN_PointNet()

        else:
            raise Exception("Not implemented")

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)
        if self.emb_nn_name == "pointnet2ori":
            points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points
        else:
            points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points

        return points_embedding


class EquivariantFeatureEmbeddingNetworkTwin(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(EquivariantFeatureEmbeddingNetworkTwin, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)

        else:
            raise Exception("Not implemented")

    def forward(self, *input):
        points = input[0].permute(0, 2, 1)  # B,3,num_points
        transformed_points = input[1].permute(0, 2, 1)
        points_dmean = points - points.mean(dim=2, keepdim=True)
        transformed_points_dmean = transformed_points - transformed_points.mean(
            dim=2, keepdim=True
        )
        points_embedding = self.emb_nn(points_dmean)
        transformed_points_embedding = self.emb_nn(transformed_points_dmean)

        return points_embedding, transformed_points_embedding


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    # src, dst (num_dims, num_points)
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)
    distances = (
        -torch.sum(src**2, dim=0, keepdim=True).transpose(1, 0).contiguous()
        - inner
        - torch.sum(dst**2, dim=0, keepdim=True)
    )
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(
            self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        )


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(
            self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous())
            .transpose(2, 1)
            .contiguous()
        )


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512, input_dims=3):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
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
        super(Transformer, self).__init__()
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


class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        batch_size = action_points.size(0)

        d_k = action_embedding.size(1)
        scores = torch.matmul(
            action_embedding.transpose(2, 1).contiguous(), anchor_embedding
        ) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        action_corr = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())

        action_centered = action_points - action_points.mean(dim=2, keepdim=True)

        action_corr_centered = action_corr - action_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(
            action_centered, action_corr_centered.transpose(2, 1).contiguous()
        )

        U, S, V = [], [], []
        R = []

        for i in range(action_points.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
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

        t = torch.matmul(
            -R, action_points.mean(dim=2, keepdim=True)
        ) + action_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


class PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(
                nn.Conv1d(layer_dims[j], layer_dims[j + 1], kernel_size=1, bias=False)
            )
            norms.append(nn.BatchNorm1d(layer_dims[j + 1]))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        for bn, conv in zip(self.norms, self.convs):
            x = F.relu(bn(conv(x)))
        return x


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
        super(MLPHead, self).__init__()

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


class ResidualMLPHeadRefinement(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512, pred_weight=True):
        super(ResidualMLPHeadRefinement, self).__init__()

        self.emb_dims = emb_dims
        self.pred_weight = pred_weight
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, 64, 64, 64, 128, 512]),
            nn.Conv1d(512, 3, kernel_size=1, bias=False),
        )
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

    def forward(self, *input, scores=None, return_flow_component=False):
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

        flow = residual_flow + corr_flow
        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return corr_flow_weight


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
        # import pdb
        # pdb.set_trace()
        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        if self.residual_on:
            # print("residual flow")
            flow = residual_flow + corr_flow
        else:
            # print("no residual flow")
            flow = corr_flow

        if self.pred_weight:
            # print("ResidualMLPHead: PRODUCING SVD WEIGHTS!!!!")
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

        return corr_flow_weight


class ResidualMLPHead_Scaled(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i

    Scaled for scaling the ACTION point cloud so that it matches with
    the test time relative porpotion of action clouds wrt to anchor clouds
    """

    def __init__(self, emb_dims=512, pred_weight=True, residual_on=True):
        super(ResidualMLPHead_Scaled, self).__init__()

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
        self,
        *input,
        scores=None,
        return_flow_component=False,
        return_embedding=False,
        scale_action=1.0,
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
        # import pdb
        # pdb.set_trace()
        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        residual_flow = self.proj_flow(embedding)  # B,3,N
        residual_flow = residual_flow / scale_action
        if self.residual_on:
            # print("residual flow")
            flow = residual_flow + corr_flow
        else:
            # print("no residual flow")
            flow = corr_flow

        if self.pred_weight:
            # print("ResidualMLPHead: PRODUCING SVD WEIGHTS!!!!")
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

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


class ResidualMLPHead_Identity(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_Identity, self).__init__()

        self.emb_dims = emb_dims + 3
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                ]
            ),
            nn.Conv1d(self.emb_dims // 8, 3, kernel_size=1, bias=False),
        )
        self.residualhead = ResidualMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_embedding_tf = input[0]
        corr_flow = self.residualhead(*input)  # B,3,N
        embedding = torch.concat(
            [action_embedding_tf, corr_flow], dim=1
        )  # B,emb_dis*2,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        return residual_flow + corr_flow


class ResidualMLPHead_V1(nn.Module):
    """
    ResidualMLP Head Variant 1 with flow calcuated as:
    v_i = f(\phi_i,\tilde{\phi}_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_V1, self).__init__()

        self.emb_dims = emb_dims * 2
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                    self.emb_dims // 16,
                ]
            ),
            nn.Conv1d(self.emb_dims // 16, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding_tf = input[0]
        anchor_embedding_tf = input[1]
        action_points = input[2]
        anchor_points = input[3]
        action_embedding = input[4]
        anchor_embedding = input[5]
        if scores is None:
            action_embedding = input[4]
            anchor_embedding = input[5]

            action_query = input[4]
            anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(
                d_k
            )  # B, N, N (N=number of points, 1024 cur)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = torch.concat(
            [action_embedding, action_embedding_tf], dim=1
        )  # B,emb_dis*2,N
        residual_flow = self.proj_flow(embedding)  # B,3,N
        return residual_flow + corr_flow


class ResidualMLPHead_V2(nn.Module):
    """
    ResidualMLP Head Variant 1 with flow calcuated as:
    v_i = f(\phi_i,\tilde{y}_i) + \tilde{y}_i -x_i
    \tilde_phi_i -> action_embedding
    \phi_i -> action_embedding_tf

    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_V2, self).__init__()

        self.emb_dims = emb_dims + 3
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                ]
            ),
            nn.Conv1d(self.emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding_tf = input[0]
        anchor_embedding_tf = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            action_embedding = input[4]
            anchor_embedding = input[5]
            action_query = input[4]
            anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(
                d_k
            )  # B, N, N (N=number of points, 1024 cur)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        tilde_y = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = tilde_y - action_points

        embedding = torch.concat([action_embedding_tf, tilde_y], dim=1)  # B,emb_dis*2,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        return residual_flow + corr_flow


class ResidualMLPHead_V3(nn.Module):
    """
    ResidualMLP Head Variant 1 with flow calcuated as:
    v_i = f(\phi_i,\tilde{y}_i) + \tilde{y}_i -x_i
    \tilde_phi_i -> action_embedding
    \phi_i -> action_embedding_tf

    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_V3, self).__init__()
        self.emb_dims = emb_dims + 3
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                ]
            ),
            nn.Conv1d(self.emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding_tf = input[0]
        anchor_embedding_tf = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            action_embedding = input[4]
            anchor_embedding = input[5]
            action_query = input[4]
            anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(
                d_k
            )  # B, N, N (N=number of points, 1024 cur)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = torch.concat(
            [action_embedding_tf, corr_flow], dim=1
        )  # B,emb_dis*2,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        return residual_flow


class ResidualMLPHead_V4(nn.Module):
    """
    ResidualMLP Head Variant 1 with flow calcuated as:
    v_i = f(\phi_i,\tilde{y}_i) + \tilde{y}_i -x_i
    \tilde_phi_i -> action_embedding
    \phi_i -> action_embedding_tf

    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_V4, self).__init__()
        self.emb_dims = emb_dims + 3
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                ]
            ),
            nn.Conv1d(self.emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding_tf = input[0]
        anchor_embedding_tf = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            action_embedding = input[4]
            anchor_embedding = input[5]
            action_query = input[4]
            anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(
                d_k
            )  # B, N, N (N=number of points, 1024 cur)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())

        embedding = torch.concat(
            [action_embedding_tf, corr_points], dim=1
        )  # B,emb_dis*2,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        return residual_flow


class ResidualMLPHead_Correspondence(nn.Module):
    """
    ResidualMLP Head Variant 1 with flow calcuated as:
    v_i = f(\phi_i,\tilde{y}_i) + \tilde{y}_i -x_i
    \tilde_phi_i -> action_embedding
    \phi_i -> action_embedding_tf

    """

    def __init__(self, emb_dims=512):
        super(ResidualMLPHead_Correspondence, self).__init__()

        self.emb_dims = emb_dims + 3
        self.proj_flow = nn.Sequential(
            PointNet(
                [
                    self.emb_dims,
                    self.emb_dims // 2,
                    self.emb_dims // 4,
                    self.emb_dims // 8,
                ]
            ),
            nn.Conv1d(self.emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding_tf = input[0]
        anchor_embedding_tf = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            action_embedding = input[4]
            anchor_embedding = input[5]
            action_query = input[4]
            anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(
                d_k
            )  # B, N, N (N=number of points, 1024 cur)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)
        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        return corr_flow


class DirectFlow(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(DirectFlow, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims)
        self.head_action = MLPHead(emb_dims=emb_dims)
        self.head_anchor = MLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        action_embedding_tf, anchor_embedding_tf = self.transformer(
            action_embedding, anchor_embedding
        )

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        flow_action = self.head_action(action_embedding_tf).permute(0, 2, 1)

        if self.cycle:
            flow_anchor = self.head_anchor(anchor_embedding_tf).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


# TODO: to be finished


class CorrespondenceFlow_DiffEmbMLP(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        inital_sampling_ratio=0.2,
        center_feature=True,
    ):
        super(CorrespondenceFlow_DiffEmbMLP, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle

        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")

        self.center_feature = center_feature

        self.transformer_action = MLP(emb_dims=emb_dims)
        self.transformer_anchor = MLP(emb_dims=emb_dims)
        self.head_action = CorrespondenceMLPHead(emb_dims=emb_dims)
        self.head_anchor = CorrespondenceMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
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


class CorrespondenceFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        inital_sampling_ratio=0.2,
        freeze_embnn=True,
        center_feature=True,
    ):
        super(CorrespondenceFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle

        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")

        self.center_feature = center_feature
        self.freeze_embnn = freeze_embnn

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.head_action = CorrespondenceMLPHead(emb_dims=emb_dims)
        self.head_anchor = CorrespondenceMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
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
        action_attn = action_attn.mean(dim=1)
        # with profiler.record_function("CorrespondenceMLPHead"):
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=None,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=None,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor
        return flow_action


class ResidualFlow_DiffEmb(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        inital_sampling_ratio=0.2,
    ):
        super(ResidualFlow_DiffEmb, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        action_embedding = self.emb_nn_action(action_points_dmean)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(action_embedding, anchor_embedding)

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

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
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action


class ResidualFlow_DiffEmbTransformerSymmetryBreaking(nn.Module):
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
        input_dims=3,
    ):
        super(ResidualFlow_DiffEmbTransformerSymmetryBreaking, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        self.input_dims = input_dims
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )

            self.emb_nn_anchor = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = Transformer(
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
        action_points_input = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points_input = input[1].permute(0, 2, 1)
        assert action_points_input.shape[1] in [
            3,
            4,
        ], "action_points shoulf be of shape (Batch, {3,4}. num_points), but got {}".format(
            action_points.shape
        )
        # compute action_dmean
        if action_points_input.shape[1] == 4:
            action_xyz = action_points_input[:, :3]
            anchor_xyz = anchor_points_input[:, :3]
            action_sym_cls = action_points_input[:, 3:]
            anchor_sym_cls = anchor_points_input[:, 3:]
            action_points_dmean = action_xyz - action_xyz.mean(dim=2, keepdim=True)
            anchor_points_dmean = anchor_xyz - anchor_xyz.mean(dim=2, keepdim=True)

        elif action_points_input.shape[1] == 3:
            action_points_dmean = action_points_input - action_points_input.mean(
                dim=2, keepdim=True
            )
            anchor_points_dmean = anchor_points_input - anchor_points_input.mean(
                dim=2, keepdim=True
            )
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points_input
            anchor_points_dmean = anchor_points_input
        action_points = action_points_input[:, :3]
        anchor_points = anchor_points_input[:, :3]
        if self.freeze_embnn:
            action_embedding = self.emb_nn_action(action_points_dmean).detach()
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean).detach()
        else:
            action_embedding = self.emb_nn_action(action_points_dmean)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        if action_points_input.shape[1] == 4:
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
        action_embedding_tf, action_attn = self.transformer_action(
            action_embedding, anchor_embedding
        )
        anchor_embedding_tf, anchor_attn = self.transformer_anchor(
            anchor_embedding, action_embedding
        )

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

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
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action


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
        return_attn=True,
        input_dims=3,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        self.input_dims = input_dims
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )
            self.emb_nn_anchor = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.return_attn = return_attn

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        self.transformer_anchor = Transformer(
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
        action_points_input = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points_input = input[1].permute(0, 2, 1)
        assert action_points_input.shape[1] in [
            3,
            4,
        ], "action_points should be of shape (Batch, {3,4}. num_points), but got {}".format(
            action_points.shape
        )
        # compute action_dmean
        if action_points_input.shape[1] == 4:
            action_xyz = action_points_input[:, :3]
            anchor_xyz = anchor_points_input[:, :3]
            action_sym_cls = action_points_input[:, 3:]
            anchor_sym_cls = anchor_points_input[:, 3:]
            action_xyz_dmean = action_xyz - action_xyz.mean(dim=2, keepdim=True)
            anchor_xyz_dmean = anchor_xyz - anchor_xyz.mean(dim=2, keepdim=True)
            action_points_dmean = torch.cat([action_xyz_dmean, action_sym_cls], axis=1)
            anchor_points_dmean = torch.cat([anchor_xyz_dmean, anchor_sym_cls], axis=1)

        elif action_points_input.shape[1] == 3:
            action_points_dmean = action_points_input - action_points_input.mean(
                dim=2, keepdim=True
            )
            anchor_points_dmean = anchor_points_input - anchor_points_input.mean(
                dim=2, keepdim=True
            )
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points_input
            anchor_points_dmean = anchor_points_input
        action_points = action_points_input[:, :3]
        anchor_points = anchor_points_input[:, :3]
        if self.freeze_embnn:
            action_embedding = self.emb_nn_action(action_points_dmean).detach()
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean).detach()
        else:
            action_embedding = self.emb_nn_action(action_points_dmean)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        if self.return_attn:
            action_embedding_tf, action_attn = self.transformer_action(
                action_embedding, anchor_embedding
            )
            anchor_embedding_tf, anchor_attn = self.transformer_anchor(
                anchor_embedding, action_embedding
            )
        else:
            action_embedding_tf = self.transformer_action(
                action_embedding, anchor_embedding
            )
            anchor_embedding_tf = self.transformer_anchor(
                anchor_embedding, action_embedding
            )
            action_attn = None
            anchor_attn = None

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf
        if action_attn is not None:
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
            if anchor_attn is not None:
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
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action


class ResidualFlow_DiffEmbTransformer_Scaled(nn.Module):
    """
    Scaled for scaling the ACTION point cloud so that it matches with
    the test time relative porpotion of action clouds wrt to anchor clouds

    Diff from ResidualFlow_DiffEmbTransformer:
        Additional Arguments
            - scale_action
            - scale_anchor
    """

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
    ):
        super(ResidualFlow_DiffEmbTransformer_Scaled, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.head_action = ResidualMLPHead_Scaled(
            emb_dims=emb_dims,
            pred_weight=self.pred_weight,
            residual_on=self.residual_on,
        )
        self.head_anchor = ResidualMLPHead_Scaled(
            emb_dims=emb_dims,
            pred_weight=self.pred_weight,
            residual_on=self.residual_on,
        )

    def forward(self, *input, scale_action=1.0, scale_anchor=1.0):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points

        action_points_dmean = scale_action * action_points_dmean
        anchor_points_dmean = scale_anchor * anchor_points_dmean

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

        action_attn = action_attn.mean(dim=1)
        if self.return_flow_component:
            flow_output_action = self.head_action(
                action_embedding_tf,
                anchor_embedding_tf,
                action_points,
                anchor_points,
                scores=action_attn,
                return_flow_component=self.return_flow_component,
                scale_action=scale_action,
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
                scale_action=scale_action,
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
                    scale_action=scale_anchor,
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
                    scale_action=scale_anchor,
                ).permute(0, 2, 1)
            if self.return_flow_component:
                return {
                    "flow_action": flow_action,
                    "flow_anchor": flow_anchor,
                    "residual_flow_action": residual_flow_action,
                    "residual_flow_anchor": residual_flow_anchor,
                    "corr_flow_action": corr_flow_action,
                    "corr_flow_anchor": corr_flow_anchor,
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action


class ResidualFlow_DiffEmbTransformerVis(nn.Module):
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
    ):
        super(ResidualFlow_DiffEmbTransformerVis, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = Transformer(
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
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
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
                }
            else:
                return flow_action, flow_anchor, action_embedding, anchor_embedding
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action, action_embedding, anchor_embedding


class ResidualFlow(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        inital_sampling_ratio=0.2,
    ):
        super(ResidualFlow, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn = Pointnet2Dense(emb_dims, params=params_action)
        elif emb_nn == "pointnet2ori":
            self.emb_nn = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        action_embedding = self.emb_nn(action_points_dmean)
        anchor_embedding = self.emb_nn(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(action_embedding, anchor_embedding)

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

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
                }
            else:
                return flow_action, flow_anchor
        if self.return_flow_component:
            return {
                "flow_action": flow_action,
                "residual_flow_action": residual_flow_action,
                "corr_flow_action": corr_flow_action,
            }
        else:
            return flow_action


class PartSegMaskRefinement(nn.Module):
    def __init__(
        self, emb_nn="dgcnn", center_feature=True, pred_weight=False, variant_type=0
    ):
        super(PartSegMaskRefinement, self).__init__()
        """
        variant_type:
            0, 3d vector flow per point, use flow for action and anchor points (dualflow2pose)
            1, 3d vector flow per point, use flow of action points only (flow2pose)
            2, 6d vector flow per point, use flow of action and anchor points (dualflow2pose)
        """
        self.pred_weight = pred_weight
        self.variant_type = variant_type
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "pointnet2ori":
            if self.pred_weight:
                if self.variant_type == 0 or self.variant_type == 1:
                    self.emb_nn = pointnet2(output_dim=4, input_dim=4)
                elif self.variant_type == 2:
                    self.emb_nn = pointnet2(output_dim=8, input_dim=4)
            else:
                if self.variant_type == 0 or self.variant_type == 1:
                    self.emb_nn = pointnet2(output_dim=3, input_dim=4)
                elif self.variant_type == 2:
                    self.emb_nn = pointnet2(output_dim=6, input_dim=4)

        elif emb_nn == "dgcnn":
            if self.pred_weight:
                if self.variant_type == 0 or self.variant_type == 1:
                    self.emb_nn = DGCNN(emb_dims=4)
                elif self.variant_type == 2:
                    raise Exception("Not implemented")
            else:
                if self.variant_type == 0 or self.variant_type == 1:
                    self.emb_nn = DGCNN(emb_dims=3)
                elif self.variant_type == 2:
                    raise Exception("Not implemented")
        elif emb_nn == "vn":
            self.emb_nn = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.center_feature = center_feature

    def forward(self, *input):
        points = input[0].permute(0, 2, 1)  # B,3,num_points
        labels = input[1].permute(0, 2, 1)  # B,1,num_points

        # Mean center point cloud before point extrator
        if self.center_feature:
            points_centered = points - points.mean(dim=2, keepdim=True)
        else:
            points_centered = points

        xyzlabel = torch.cat([points_centered, labels], dim=1)  # B, 4, num_point

        flow = self.emb_nn(xyzlabel).permute(0, 2, 1)
        # B, num_points, 3 if variant_type = {0,1}
        # B, num_points, 6 if variant_type = {2}
        return flow


class ResidualFlowRefinement(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        center_feature=True,
        pred_weight=True,
        inital_sampling_ratio=0.2,
    ):
        super(ResidualFlowRefinement, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "pointnet2ben":
            params = PN2DenseParams()
            params.sa1.ratio = inital_sampling_ratio
            self.emb_nn = Pointnet2Dense(emb_dims, params=params)
        elif emb_nn == "pointnet2ori":
            self.emb_nn = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == "vn":
            self.emb_nn = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.center_feature = center_feature
        self.pred_weight = pred_weight

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHeadRefinement(
            emb_dims=emb_dims, pred_weight=self.pred_weight
        )
        self.head_anchor = ResidualMLPHeadRefinement(
            emb_dims=emb_dims, pred_weight=self.pred_weight
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
        action_embedding = self.emb_nn(action_points_dmean)
        anchor_embedding = self.emb_nn(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(action_embedding, anchor_embedding)

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)

        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)

            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)

            return flow_action, flow_anchor

        return flow_action


def scale_coords(src, scale_len):
    """
    src of shape: B,3,N
    """
    src_zeromined = src - src.min()
    src_normalized = src_zeromined / src_zeromined.max()
    src_scaled = (src_normalized * scale_len).int()
    return src_scaled


class ResidualFlow_PE(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn", scale_len=50):
        super(ResidualFlow_PE, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")
        self.scale_len = scale_len
        self.positional_encoding = PositionalEncoding3D(channels=self.emb_dims)
        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        # TODO:
        # Use coordinaters of each point to get positional_endocings and add to action_embedding/anchor_embedding
        action_embedding = self.emb_nn(action_points)  # B,512,N
        anchor_embedding = self.emb_nn(anchor_points)

        src_embedding_tf, tgt_embedding_tf, action_attn, anchor_attn = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N
        shape_tensor = torch.zeros(
            action_points.shape[0],
            self.scale_len,
            self.scale_len,
            self.scale_len,
            self.emb_dims,
        )

        pe = self.positional_encoding(shape_tensor)  # B,100,100,100,512
        src_scaled = scale_coords(action_points, self.scale_len)  # B,3,N
        tgt_scaled = scale_coords(anchor_points, self.scale_len)  # B,3,N

        src_index = torch.transpose(src_scaled, 1, 2)  # B,N,3
        tgt_index = torch.transpose(tgt_scaled, 1, 2)  # B,N,3

        # for i in range(src_index.shape[1]):
        l = []
        for i in range(action_embedding.shape[-1]):
            perpt_idx = src_index[:, i].long()  # B,1,3

            emb = pe[
                torch.arange(src_index.shape[0]),
                perpt_idx[:, 0],
                perpt_idx[:, 1],
                perpt_idx[:, 2],
                :,
            ].unsqueeze(
                1
            )  # B,512
            l.append(emb)

        src_pe = torch.cat(l, 1)

        src_embedding_tf = action_embedding + src_embedding_tf
        tgt_embedding_tf = anchor_embedding + tgt_embedding_tf
        action_attn = action_attn.mean(dim=1)
        flow_ab = self.head_src(
            src_embedding_tf,
            tgt_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_ba = self.head_tgt(
                tgt_embedding_tf,
                src_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_ab, flow_ba

        return flow_ab


class ResidualFlow_V1(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_V1, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_V1(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_V1(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            action_embedding,
            anchor_embedding,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                anchor_embedding,
                action_embedding,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


class ResidualFlow_V2(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_V2, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_V2(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_V2(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


class ResidualFlow_V3(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_V3, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_V3(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_V3(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


class ResidualFlow_V4(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_V4, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_V4(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_V4(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


class ResidualFlow_Correspondence(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_Correspondence, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_Correspondence(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_Correspondence(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


class ResidualFlow_Identity(nn.Module):
    def __init__(self, emb_dims=512, cycle=True, emb_nn="dgcnn"):
        super(ResidualFlow_Identity, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            self.emb_nn = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")

        self.transformer = Transformer(emb_dims=emb_dims, return_attn=True)
        self.head_action = ResidualMLPHead_Identity(emb_dims=emb_dims)
        self.head_anchor = ResidualMLPHead_Identity(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)
        anchor_points = input[1].permute(0, 2, 1)
        action_embedding = self.emb_nn(action_points)
        anchor_embedding = self.emb_nn(anchor_points)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
        ) = self.transformer(
            action_embedding, anchor_embedding
        )  # tilde_phi, phi are both B,512,N

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        action_attn = action_attn.mean(dim=1)
        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            action_embedding,
            anchor_embedding,
        ).permute(0, 2, 1)

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            ).permute(0, 2, 1)
            return flow_action, flow_anchor

        return flow_action


max_id = -1
floor_id = 0

table_id = 2
mug_id = 3
rack_id = 16777218

base_id = 1
link_0_id = 16777217
link_1_id = 33554433
link_2_id = 50331649
link_3_id = 83886081
link_4_id = 100663297

gripper_0_id = 117440513
gripper_1_id = 150994945
finger_0_id = 167772161
finger_1_id = 184549377
gripper_ids = [gripper_0_id, gripper_1_id, finger_0_id, finger_1_id]


def get_clouds(cams, occlusion=False):
    cloud_classes = []
    cloud_points = []
    cloud_colors = []
    if occlusion:
        cam_indices = list(np.random.choice(len(cams.cams), 3, replace=False))
    else:
        cam_indices = list(np.arange(len(cams.cams)))
    for i, cam in enumerate(cams.cams):
        if i in cam_indices:
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            points, colors = cam.get_pcd(
                in_world=True,
                rgb_image=rgb,
                depth_image=depth,
                depth_min=0.0,
                depth_max=np.inf,
            )

            cloud_points.append(points)
            cloud_colors.append(colors)
            cloud_classes.append(seg.flatten())

    return cloud_points, cloud_colors, cloud_classes


def get_object_clouds(cams, ids=[mug_id, rack_id, gripper_ids], occlusion=False):
    cloud_classes = []
    cloud_points = []
    cloud_colors = []
    if occlusion:
        cam_indices = list(np.random.choice(len(cams.cams), 3, replace=False))
    else:
        cam_indices = list(np.arange(len(cams.cams)))
    for i, cam in enumerate(cams.cams):
        if i in cam_indices:
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            points, colors = cam.get_pcd(
                in_world=True,
                rgb_image=rgb,
                depth_image=depth,
                depth_min=0.0,
                depth_max=np.inf,
            )
            classes = seg.flatten()  # flat_seg
            for j, obj_ids in enumerate(ids):
                mask = np.isin(classes, obj_ids)
                cloud_points.append(points[mask])
                cloud_colors.append(colors[mask])
                cloud_classes.append(j * np.ones(mask.sum()))

    cloud_points = np.concatenate(cloud_points, axis=0)
    cloud_colors = np.concatenate(cloud_colors, axis=0)
    cloud_classes = np.concatenate(cloud_classes, axis=0)
    return cloud_points, cloud_colors, cloud_classes


def get_object_clouds_from_demo_npz(grasp_data, ids=[mug_id, rack_id, gripper_ids]):
    print("-------------inside get_object_clouds_from_demo_npz()-------------")
    grasp_data_dict = dict(grasp_data)
    cloud_points = grasp_data_dict["object_pointcloud"]
    seg = grasp_data_dict["seg"]  # 4,1
    obj_pose_world = grasp_data_dict["obj_pose_world"]
    l_seg = []
    for i in range(4):
        l_seg.append(seg[i, 0])
    l_seg_cat = np.concatenate(l_seg)
    print("cloud_points.shape:{}".format(cloud_points.shape))
    print("l_seg_cat.shape:{}".format(l_seg_cat.shape))
    print("obj_pose_world.shape:{}".format(obj_pose_world.shape))

    cloud_classes = []
    cloud_points = []
    cloud_colors = []

    cloud_points = np.concatenate(cloud_points, axis=0)
    cloud_colors = np.concatenate(cloud_colors, axis=0)
    cloud_classes = np.concatenate(cloud_classes, axis=0)

    return cloud_points, cloud_colors, cloud_classes


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
    N,
    rot_var=np.pi / 180 * 5,
    trans_var=0.1,
    device=None,
    fix_random=False,
    return_components=False,
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
    if return_components:
        return R, t
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
    temperature=1,
):
    """
    @param xyz: (batch, num_points, 3)
    @param flow: (batch, num_points,3)
    @param weights: (batch, num_points)
    @param normalization_scehme: {'l1, 'softmax'}
    @param x: flow prediction
    @return pred_T_action: SE(3) transformation from xyz to the other point cloud
    """
    xyz = xyz[:, :, :3]
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
    xyz_centered = xyz - xyz_mean

    flow_mean = (w * flow).sum(dim=1, keepdims=True)
    flow_centered = flow - flow_mean

    xyz_trans = xyz_centered + flow_centered
    X = torch.bmm(xyz_centered.transpose(-2, -1), w * xyz_trans)

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
    """
    xyz_src, batch, num_points, {3,4}
    """

    xyz_src = xyz_src[:, :, :3]
    xyz_tgt = xyz_tgt[:, :, :3]
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

    import pdb

    pdb.set_trace()
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


mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()

# color scheme from https://personal.sron.nl/~pault/
color_scheme = {}
color_scheme["blue"] = np.array([68, 119, 170])
color_scheme["cyan"] = np.array([102, 204, 238])
color_scheme["green"] = np.array([34, 136, 51])
color_scheme["yellow"] = np.array([204, 187, 68])
color_scheme["red"] = np.array([238, 102, 119])
color_scheme["purple"] = np.array([170, 51, 119])
color_scheme["orange"] = np.array([238, 119, 51])


def get_color(tensor_list, color_list, axis=False):
    """
    @param tensor_list: list of tensors of shape (num_points, 3)
    @param color_list: list of strings of color names that should be within color_scheme.keys(), eg, 'red', 'blue'
    @return points_color_stacked: numpy array of shape (num_points*len(tensor_list), 6)
    """

    stacked_list = []
    assert len(tensor_list) == len(
        color_list
    ), "len(tensor_list) should match len(color_list)"
    for i in range(len(tensor_list)):
        tensor = tensor_list[i].detach().cpu().numpy()
        color = color_list[i]
        assert len(tensor.shape) == 2, "tensor should be of shape of (num_points, 3)"
        assert (
            tensor.shape[-1] == 3
        ), "tensor.shape[-1] should be of shape 3, for point coordinates"
        assert (
            color in color_scheme.keys()
        ), "passed in color {} is not in the available color scheme, go to utils/color_utils.py to add any".format(
            color
        )

        color_tensor = torch.from_numpy(color_scheme[color]).unsqueeze(0)  # 1,3
        N = tensor.shape[0]
        color_tensor = (
            torch.repeat_interleave(color_tensor, N, dim=0).detach().cpu().numpy()
        )  # N,3

        points_color = np.concatenate((tensor, color_tensor), axis=-1)  # num_points, 6

        stacked_list.append(points_color)
    points_color_stacked = np.concatenate(
        stacked_list, axis=0
    )  # num_points*len(tensor_list), 6
    if axis:
        axis_pts = create_axis()
        points_color_stacked = np.concatenate([points_color_stacked, axis_pts], axis=0)

    return points_color_stacked


def get_color_sym_dist(tensor_list, color_list):
    """
    @param tensor_list: list of tensors of shape (num_points, 3)
    @param color_list: list of strings of color names that should be within color_scheme.keys(), eg, 'red', 'blue'
    @return points_color_stacked: numpy array of shape (num_points*len(tensor_list), 6)
    """

    stacked_list = []
    assert len(tensor_list) == len(
        color_list
    ), "len(tensor_list) should match len(color_list)"

    for i in range(len(tensor_list)):
        tensor = tensor_list[i].detach().cpu().numpy()
        if color_list[i] == None:
            color = color_scheme["blue"]
            color_tensor = torch.from_numpy(color).unsqueeze(0)  # 1,3
            N = tensor.shape[0]
            color_tensor = (
                torch.repeat_interleave(color_tensor, N, dim=0).detach().cpu().numpy()
            )  # N,3

        else:
            color_tensor = color_list[i]
        assert len(tensor.shape) == 2, "tensor should be of shape of (num_points, 3)"
        assert (
            tensor.shape[-1] == 3
        ), "tensor.shape[-1] should be of shape 3, for point coordinates"
        assert (
            len(color_tensor.shape) == 2
        ), "color_tensor should be of shape of (num_points, 3)"
        assert (
            color_tensor.shape[-1] == 3
        ), "color_tensor.shape[-1] should be of shape 3, for point coordinates"

        points_color = np.concatenate((tensor, color_tensor), axis=-1)  # num_points, 6

        stacked_list.append(points_color)

    points_color_stacked = np.concatenate(
        stacked_list, axis=0
    )  # num_points*len(tensor_list), 6

    return points_color_stacked


def create_axis(length=1.0, num_points=100):
    pts = np.linspace(0, length, num_points)
    x_axis_pts = np.stack([pts, np.zeros(num_points), np.zeros(num_points)], axis=1)
    y_axis_pts = np.stack([np.zeros(num_points), pts, np.zeros(num_points)], axis=1)
    z_axis_pts = np.stack([np.zeros(num_points), np.zeros(num_points), pts], axis=1)

    x_axis_clr = np.tile([255, 0, 0], [num_points, 1])
    y_axis_clr = np.tile([0, 255, 0], [num_points, 1])
    z_axis_clr = np.tile([0, 0, 255], [num_points, 1])

    x_axis = np.concatenate([x_axis_pts, x_axis_clr], axis=1)
    y_axis = np.concatenate([y_axis_pts, y_axis_clr], axis=1)
    z_axis = np.concatenate([z_axis_pts, z_axis_clr], axis=1)

    pts = np.concatenate([x_axis, y_axis, z_axis], axis=0)

    return pts


class PointCloudTrainingModule(pl.LightningModule):
    def __init__(self, model=None, lr=1e-3, image_log_period=500):
        super().__init__()
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.global_val_step = 0

    def module_step(self, batch, batch_idx):
        raise NotImplementedError("module_step must be implemented by child class")
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def training_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            key: val,
                            "trainer/global_step": self.global_step,
                        }
                    )
                else:
                    self.logger.log_image(
                        key,
                        images=[val],  # self.global_step
                    )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log("val_" + key, val)

        if (self.global_val_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            "val_" + key: val,
                            "trainer/global_step": self.global_val_step,
                        }
                    )
                else:
                    self.logger.log_image(
                        "val_" + key,
                        images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            "test_" + key: val,
                        }
                    )
                else:
                    self.logger.log_image("test_" + key, val)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer


class EquivarianceTestingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model=None,
        lr=1e-3,
        image_log_period=500,
        action_weight=1,
        anchor_weight=1,
        smoothness_weight=0.1,
        rotation_weight=0,
        chamfer_weight=10000,
        point_loss_type=0,
        return_flow_component=False,
        weight_normalize="l1",
        softmax_temperature=1,
        loop=3,
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
        )
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.smoothness_weight = smoothness_weight
        self.rotation_weight = rotation_weight
        self.chamfer_weight = chamfer_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize
        # 0 for mse loss, 1 for chamfer distance, 2 for mse loss + chamfer distance
        self.point_loss_type = point_loss_type
        self.return_flow_component = return_flow_component
        self.loop = loop
        self.softmax_temperature = softmax_temperature

    def action_centered(self, points_action, points_anchor):
        """
        @param points_action, (1,num_points,3)
        @param points_anchor, (1,num_points,3)
        """
        points_action_mean = points_action.clone().mean(axis=1)
        points_action_mean_centered = points_action - points_action_mean
        points_anchor_mean_centered = points_anchor - points_action_mean

        return (
            points_action_mean_centered,
            points_anchor_mean_centered,
            points_action_mean,
        )

    def get_transform(self, points_trans_action, points_trans_anchor):
        for i in range(self.loop):
            x_action, x_anchor = self.model(points_trans_action, points_trans_anchor)
            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]
            ans_dict = self.predict(
                x_action=x_action,
                x_anchor=x_anchor,
                points_trans_action=points_trans_action,
                points_trans_anchor=points_trans_anchor,
            )
            if i == 0:
                pred_T_action = ans_dict["pred_T_action"]
            else:
                pred_T_action = pred_T_action.compose(
                    T_trans.inverse()
                    .compose(ans_dict["pred_T_action"])
                    .compose(T_trans)
                )
                ans_dict["pred_T_action"] = pred_T_action
            pred_points_action = ans_dict["pred_points_action"]
            (
                points_trans_action,
                points_trans_anchor,
                points_action_mean,
            ) = self.action_centered(pred_points_action, points_trans_anchor)
            T_trans = pure_translation_se3(
                1, points_action_mean.squeeze(), device=points_trans_action.device
            )

        return ans_dict

    def predict(self, x_action, x_anchor, points_trans_action, points_trans_anchor):
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        pred_T_action = dualflow2pose(
            xyz_src=points_trans_action,
            xyz_tgt=points_trans_anchor,
            flow_src=pred_flow_action,
            flow_tgt=pred_flow_anchor,
            weights_src=pred_w_action,
            weights_tgt=pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
            temperature=self.softmax_temperature,
        )
        pred_points_action = pred_T_action.transform_points(points_trans_action)

        return {
            "pred_T_action": pred_T_action,
            "pred_points_action": pred_points_action,
        }

    def compute_loss(
        self,
        x_action,
        x_anchor,
        batch,
        log_values={},
        loss_prefix="",
        pred_points_action=None,
    ):
        if "T0" in batch.keys():
            T0 = Transform3d(matrix=batch["T0"])
        if "T1" in batch.keys():
            T1 = Transform3d(matrix=batch["T1"])

        if pred_points_action == None:
            points_trans_action = batch["points_action_trans"]
        else:
            points_trans_action = pred_points_action
        points_trans_anchor = batch["points_anchor_trans"]

        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        pred_T_action = dualflow2pose(
            xyz_src=points_trans_action,
            xyz_tgt=points_trans_anchor,
            flow_src=pred_flow_action,
            flow_tgt=pred_flow_anchor,
            weights_src=pred_w_action,
            weights_tgt=pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
            temperature=self.softmax_temperature,
        )
        pred_R_max, pred_R_min, pred_R_mean = get_degree_angle(pred_T_action)
        pred_t_max, pred_t_min, pred_t_mean = get_translation(pred_T_action)
        induced_flow_action = (
            pred_T_action.transform_points(points_trans_action) - points_trans_action
        ).detach()
        pred_points_action = pred_T_action.transform_points(points_trans_action)

        if "T0" in batch.keys():
            R0_max, R0_min, R0_mean = get_degree_angle(T0)
            t0_max, t0_min, t0_mean = get_translation(T0)
            log_values[loss_prefix + "R0_mean"] = R0_mean
            log_values[loss_prefix + "R0_max"] = R0_max
            log_values[loss_prefix + "R0_min"] = R0_min
            log_values[loss_prefix + "t0_mean"] = t0_mean
            log_values[loss_prefix + "t0_max"] = t0_max
            log_values[loss_prefix + "t0_min"] = t0_min
        if "T1" in batch.keys():
            R1_max, R1_min, R1_mean = get_degree_angle(T1)
            t1_max, t1_min, t1_mean = get_translation(T1)

            log_values[loss_prefix + "R1_mean"] = R1_mean
            log_values[loss_prefix + "R1_max"] = R1_max
            log_values[loss_prefix + "R1_min"] = R1_min

            log_values[loss_prefix + "t1_mean"] = t1_mean
            log_values[loss_prefix + "t1_max"] = t1_max
            log_values[loss_prefix + "t1_min"] = t1_min

        log_values[loss_prefix + "pred_R_max"] = pred_R_max
        log_values[loss_prefix + "pred_R_min"] = pred_R_min
        log_values[loss_prefix + "pred_R_mean"] = pred_R_mean

        log_values[loss_prefix + "pred_t_max"] = pred_t_max
        log_values[loss_prefix + "pred_t_min"] = pred_t_min
        log_values[loss_prefix + "pred_t_mean"] = pred_t_mean
        loss = 0

        return loss, log_values, pred_points_action

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            pred_w = torch.sigmoid(x[:, :, 3])
        else:
            pred_w = None
        return pred_flow, pred_w

    def module_step(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]

        log_values = {}
        pred_points_action = points_trans_action
        for i in range(self.loop):
            x_action, x_anchor = self.model(pred_points_action, points_trans_anchor)
            loss, log_values, pred_points_action = self.compute_loss(
                x_action.detach(),
                x_anchor.detach(),
                batch,
                log_values=log_values,
                loss_prefix=str(i),
                pred_points_action=pred_points_action,
            )
        return 0, log_values

    def visualize_results(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        self.predicted_action_list = []
        res_images = {}

        pred_points_action = points_trans_action
        for i in range(self.loop):
            x_action, x_anchor = self.model(pred_points_action, points_trans_anchor)

            ans_dict = self.predict(
                x_action=x_action,
                x_anchor=x_anchor,
                points_trans_action=pred_points_action,
                points_trans_anchor=points_trans_anchor,
            )

            demo_points_apply_action_transform = get_color(
                tensor_list=[
                    pred_points_action[0],
                    ans_dict["pred_points_action"][0],
                    points_trans_anchor[0],
                ],
                color_list=["blue", "green", "red"],
            )
            res_images["demo_points_apply_action_transform_" + str(i)] = wandb.Object3D(
                demo_points_apply_action_transform
            )
            pred_points_action = ans_dict["pred_points_action"]
            self.predicted_action_list.append(pred_points_action)

        transformed_input_points = get_color(
            tensor_list=[points_trans_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["input_points"] = wandb.Object3D(transformed_input_points)

        demo_points_apply_action_transform = get_color(
            tensor_list=[self.predicted_action_list[-1][0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["demo_points_apply_action_transform"] = wandb.Object3D(
            demo_points_apply_action_transform
        )

        # for i in range(len(self.predicted_action_list)):
        #     demo_points_apply_action_transform = get_color(
        #         tensor_list=[self.predicted_action_list[i][0], points_trans_anchor[0]], color_list=['blue', 'red'])
        #     res_images['demo_points_apply_action_transform_'+str(i)] = wandb.Object3D(
        #         demo_points_apply_action_transform)

        return res_images


def get_world_transform(pred_T_action_mat, obj_start_pose, point_cloud, invert=False):
    """
    pred_T_action_mat: normal SE(3) [R|t]
                                    [0|1] in object frame
    obj_start_pose: stamped_pose of obj in world frame
    """
    point_cloud_mean = point_cloud.squeeze(0).mean(axis=0).tolist()
    obj_start_pose_list = util.pose_stamped2list(obj_start_pose)
    # import pdb
    # pdb.set_trace()
    pose = util.pose_from_matrix(pred_T_action_mat)
    centering_mat = np.eye(4)
    # centering_mat[:3, 3] = -np.array(obj_start_pose_list[:3])
    centering_mat[:3, 3] = -np.array(point_cloud_mean)
    centering_pose = util.pose_from_matrix(centering_mat)
    uncentering_pose = util.pose_from_matrix(np.linalg.inv(centering_mat))

    centered_pose = util.transform_pose(
        pose_source=obj_start_pose, pose_transform=centering_pose
    )  # obj_start_pose: stamped_pose
    trans_pose = util.transform_pose(pose_source=centered_pose, pose_transform=pose)
    final_pose = util.transform_pose(
        pose_source=trans_pose, pose_transform=uncentering_pose
    )
    if invert:
        final_pose = util.pose_from_matrix(
            np.linalg.inv(util.matrix_from_pose(final_pose))
        )

    return final_pose


def load_data(num_points, clouds, classes, action_class, anchor_class):
    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()
    points_action_mean_np = points_action_np.mean(axis=0)
    points_action_np = points_action_np - points_action_mean_np

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
    points_anchor_np = points_anchor_np - points_action_mean_np
    points_anchor_mean_np = points_anchor_np.mean(axis=0)

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    points_action, points_anchor = subsample(num_points, points_action, points_anchor)

    return points_action.cuda(), points_anchor.cuda()


def load_data_raw(num_points, clouds, classes, action_class, anchor_class):
    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    points_action, points_anchor = subsample(num_points, points_action, points_anchor)
    if points_action is None:
        return None, None

    return points_action.cuda(), points_anchor.cuda()


def subsample(num_points, points_action, points_anchor):
    if points_action.shape[1] > num_points:
        points_action, _ = sample_farthest_points(
            points_action, K=num_points, random_start_point=True
        )
    elif points_action.shape[1] < num_points:
        log_info(
            f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})"
        )
        return None, None
        # raise NotImplementedError(
        #     f'Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})')

    if points_anchor.shape[1] > num_points:
        points_anchor, _ = sample_farthest_points(
            points_anchor, K=num_points, random_start_point=True
        )
    elif points_anchor.shape[1] < num_points:
        log_info(
            f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})"
        )
        return None, None
        # raise NotImplementedError(
        #     f'Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})')

    return points_action, points_anchor


@hydra.main(
    config_path="../configs/",
    config_name="eval_full_mug_standalone",
)
def main(hydra_cfg):
    txt_file_name = "{}.txt".format(hydra_cfg.eval_data_dir)
    data_dir = hydra_cfg.data_dir
    save_dir = os.path.join("./ben_trying_things", data_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    hydra_cfg.eval_data_dir = hydra_cfg.data_dir
    obj_class = hydra_cfg.object_class
    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(), obj_class + "_centered_obj_normalized"
    )

    demo_load_dir = osp.join(
        path_util.get_ndf_data(), "demos", obj_class, hydra_cfg.demo_exp
    )

    expstr = "exp--" + str(hydra_cfg.exp)
    modelstr = "model--" + str(hydra_cfg.model_path)
    seedstr = "seed--" + str(hydra_cfg.seed)
    full_experiment_name = "_".join([expstr, modelstr, seedstr])

    eval_save_dir = osp.join(
        path_util.get_ndf_eval_data(), hydra_cfg.eval_data_dir, full_experiment_name
    )
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(), hydra_cfg.model_path + ".pth"
    )

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path,
    )

    if hydra_cfg.debug:
        set_log_level("debug")
    else:
        set_log_level("info")

    robot = Robot(
        "franka",
        pb_cfg={"gui": hydra_cfg.pybullet_viz},
        arm_cfg={"self_collision": False, "seed": hydra_cfg.seed},
    )
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(hydra_cfg.seed)
    random.seed(hydra_cfg.seed)
    np.random.seed(hydra_cfg.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(
        path_util.get_ndf_config(), "eval_cfgs", hydra_cfg.config + ".yaml"
    )
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info("Config file %s does not exist, using defaults" % config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(
        path_util.get_ndf_config(), hydra_cfg.object_class + "_obj_cfg.yaml"
    )
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict["shapenet_obj_dir"]
    obj_class = global_dict["object_class"]
    eval_save_dir = global_dict["eval_save_dir"]

    eval_grasp_imgs_dir = osp.join(eval_save_dir, "grasp_imgs")
    eval_teleport_imgs_dir = osp.join(eval_save_dir, "teleport_imgs")
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)

    test_shapenet_ids = np.loadtxt(
        osp.join(path_util.get_ndf_share(), "%s_test_object_split.txt" % obj_class),
        dtype=str,
    ).tolist()
    if obj_class == "mug":
        avoid_shapenet_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
    elif obj_class == "bowl":
        avoid_shapenet_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
    elif obj_class == "bottle":
        avoid_shapenet_ids = (
            bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS
        )
    else:
        test_shapenet_ids = []

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
    preplace_horizontal_tf = util.list2pose_stamped(cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
    preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)

    if hydra_cfg.dgcnn:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type="dgcnn",
            return_features=True,
            sigmoid=True,
            acts=hydra_cfg.acts,
        ).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True
        ).cuda()

    if not hydra_cfg.random:
        checkpoint_path = global_dict["vnn_checkpoint_path"]
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        pass

    if cfg.DEMOS.PLACEMENT_SURFACE == "shelf":
        load_shelf = True
    else:
        load_shelf = False

    # get filenames of all the demo files
    demo_filenames = os.listdir(global_dict["demo_load_dir"])
    assert len(demo_filenames), (
        "No demonstrations found in path: %s!" % global_dict["demo_load_dir"]
    )

    # strip the filenames to properly pair up each demo file
    grasp_demo_filenames_orig = [
        osp.join(global_dict["demo_load_dir"], fn)
        for fn in demo_filenames
        if "grasp_demo" in fn
    ]  # use the grasp names as a reference

    place_demo_filenames = []
    grasp_demo_filenames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split("/")[-1].split("grasp_demo_")[-1]
        place_fname = osp.join(
            "/".join(fname.split("/")[:-1]), "place_demo_" + shapenet_id_npz
        )
        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
        else:
            log_warn(
                "Could not find corresponding placement demo: %s, skipping "
                % place_fname
            )

    success_list = []
    place_success_list = []
    place_success_teleport_list = []
    grasp_success_list = []

    place_fail_list = []
    place_fail_teleport_list = []
    grasp_fail_list = []

    demo_shapenet_ids = []

    # get info from all demonstrations
    demo_target_info_list = []
    demo_rack_target_info_list = []

    if hydra_cfg.n_demos > 0:
        gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
        gp_fns = random.sample(gp_fns, hydra_cfg.n_demos)
        grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
        grasp_demo_filenames, place_demo_filenames = list(grasp_demo_filenames), list(
            place_demo_filenames
        )
        log_warn("USING ONLY %d DEMONSTRATIONS" % len(grasp_demo_filenames))
        print(grasp_demo_filenames, place_demo_filenames)
    else:
        log_warn("USING ALL %d DEMONSTRATIONS" % len(grasp_demo_filenames))

    grasp_demo_filenames = grasp_demo_filenames[: hydra_cfg.num_demo]
    place_demo_filenames = place_demo_filenames[: hydra_cfg.num_demo]

    max_bb_volume = 0
    place_xq_demo_idx = 0
    grasp_data_list = []
    place_data_list = []
    demo_rel_mat_list = []

    # load all the demo data and look at objects to help decide on query points
    for i, fname in enumerate(grasp_demo_filenames):
        print("Loading demo from fname: %s" % fname)
        grasp_demo_fn = grasp_demo_filenames[i]
        place_demo_fn = place_demo_filenames[i]
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        grasp_data_list.append(grasp_data)
        place_data_list.append(place_data)

        start_ee_pose = grasp_data["ee_pose_world"].tolist()
        end_ee_pose = place_data["ee_pose_world"].tolist()
        place_rel_mat = util.get_transform(
            pose_frame_target=util.list2pose_stamped(end_ee_pose),
            pose_frame_source=util.list2pose_stamped(start_ee_pose),
        )
        place_rel_mat = util.matrix_from_pose(place_rel_mat)
        demo_rel_mat_list.append(place_rel_mat)

        if i == 0:
            (
                optimizer_gripper_pts,
                rack_optimizer_gripper_pts,
                shelf_optimizer_gripper_pts,
            ) = process_xq_data(grasp_data, place_data, shelf=load_shelf)
            (
                optimizer_gripper_pts_rs,
                rack_optimizer_gripper_pts_rs,
                shelf_optimizer_gripper_pts_rs,
            ) = process_xq_rs_data(grasp_data, place_data, shelf=load_shelf)

            if cfg.DEMOS.PLACEMENT_SURFACE == "shelf":
                print("Using shelf points")
                place_optimizer_pts = shelf_optimizer_gripper_pts
                place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
            else:
                print("Using rack points")
                place_optimizer_pts = rack_optimizer_gripper_pts
                place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs

        if cfg.DEMOS.PLACEMENT_SURFACE == "shelf":
            target_info, rack_target_info, shapenet_id = process_demo_data_shelf(
                grasp_data, place_data, cfg=None
            )
        else:
            target_info, rack_target_info, shapenet_id = process_demo_data_rack(
                grasp_data, place_data, cfg=None
            )

        if cfg.DEMOS.PLACEMENT_SURFACE == "shelf":
            rack_target_info["demo_query_pts"] = place_optimizer_pts
        demo_target_info_list.append(target_info)
        demo_rack_target_info_list.append(rack_target_info)
        demo_shapenet_ids.append(shapenet_id)

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_optimizer_pts,
        query_pts_real_shape=place_optimizer_pts_rs,
        opt_iterations=hydra_cfg.opt_iterations,
    )

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=optimizer_gripper_pts,
        query_pts_real_shape=optimizer_gripper_pts_rs,
        opt_iterations=hydra_cfg.opt_iterations,
    )
    grasp_optimizer.set_demo_info(demo_target_info_list)
    place_optimizer.set_demo_info(demo_rack_target_info_list)

    # get objects that we can use for testing
    test_object_ids = []
    shapenet_id_list = (
        [fn.split("_")[0] for fn in os.listdir(shapenet_obj_dir)]
        if obj_class == "mug"
        else os.listdir(shapenet_obj_dir)
    )
    for s_id in shapenet_id_list:
        valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids
        if hydra_cfg.only_test_ids:
            valid = valid and (s_id in test_shapenet_ids)

        if valid:
            test_object_ids.append(s_id)

    if hydra_cfg.single_instance:
        test_object_ids = [demo_shapenet_ids[0]]

    # reset
    robot.arm.reset(force_reset=True)
    robot.cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z], dist=0.9, yaw=45, pitch=-25, roll=0
    )

    cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info["pose_world"] = []
    for cam in cams.cams:
        cam_info["pose_world"].append(util.pose_from_matrix(cam.cam_ext_mat))

    # put table at right spot
    table_ori = euler2quat([0, 0, np.pi / 2])

    # this is the URDF that was used in the demos -- make sure we load an identical one
    tmp_urdf_fname = osp.join(
        path_util.get_ndf_descriptions(), "hanging/table/table_rack_tmp.urdf"
    )
    open(tmp_urdf_fname, "w").write(grasp_data["table_urdf"].item())
    table_id = robot.pb_client.load_urdf(
        tmp_urdf_fname, cfg.TABLE_POS, table_ori, scaling=cfg.TABLE_SCALING
    )

    if obj_class == "mug":
        rack_link_id = 0
        shelf_link_id = 1
    elif obj_class in ["bowl", "bottle"]:
        rack_link_id = None
        shelf_link_id = 0

    if cfg.DEMOS.PLACEMENT_SURFACE == "shelf":
        placement_link_id = shelf_link_id
    else:
        placement_link_id = rack_link_id

    def hide_link(obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    viz_data_list = []

    pl.seed_everything(hydra_cfg.seed)

    if hydra_cfg.flow_compute_type == 0:
        if hydra_cfg.diff_emb:
            if hydra_cfg.diff_transformer:
                if hydra_cfg.mlp:
                    network = CorrespondenceFlow_DiffEmbMLP(
                        emb_dims=hydra_cfg.emb_dims,
                        emb_nn=hydra_cfg.emb_nn,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                    )
                else:
                    network = ResidualFlow_DiffEmbTransformer(
                        pred_weight=hydra_cfg.pred_weight,
                        emb_nn=hydra_cfg.emb_nn,
                        return_flow_component=hydra_cfg.return_flow_component,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                        residual_on=hydra_cfg.residual_on,
                    )
            else:
                network = ResidualFlow_DiffEmb(
                    emb_nn=hydra_cfg.emb_nn,
                    return_flow_component=hydra_cfg.return_flow_component,
                    center_feature=hydra_cfg.center_feature,
                    inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                )
        else:
            network = ResidualFlow(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
                center_feature=hydra_cfg.center_feature,
            )
    elif hydra_cfg.flow_compute_type == 1:
        network = ResidualFlow_V1(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 2:
        network = ResidualFlow_V2(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 3:
        network = ResidualFlow_V3(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 4:
        network = ResidualFlow_V4(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 5:
        network = ResidualFlow_Correspondence(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == 6:
        network = ResidualFlow_Identity(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == "pe":
        network = ResidualFlow_PE(emb_nn=hydra_cfg.emb_nn)
    place_model = EquivarianceTestingModule(
        network,
        lr=hydra_cfg.lr,
        image_log_period=hydra_cfg.image_logging_period,
        weight_normalize=hydra_cfg.weight_normalize_place,
        loop=hydra_cfg.loop,
    )

    place_model.cuda()

    if hydra_cfg.checkpoint_file_place is not None:
        place_model.load_state_dict(
            torch.load(hydra_cfg.checkpoint_file_place)["state_dict"]
        )
        log_info("Model Loaded from " + str(hydra_cfg.checkpoint_file_place))
    if hydra_cfg.checkpoint_file_place_refinement is not None:
        if hydra_cfg.flow_compute_type == 0:
            if hydra_cfg.diff_emb:
                if hydra_cfg.diff_transformer:
                    if hydra_cfg.mlp:
                        network = CorrespondenceFlow_DiffEmbMLP(
                            emb_dims=hydra_cfg.emb_dims,
                            emb_nn=hydra_cfg.emb_nn,
                            center_feature=hydra_cfg.center_feature,
                            inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                        )
                    else:
                        network = ResidualFlow_DiffEmbTransformer(
                            pred_weight=hydra_cfg.pred_weight,
                            emb_nn=hydra_cfg.emb_nn,
                            return_flow_component=hydra_cfg.return_flow_component,
                            center_feature=hydra_cfg.center_feature,
                            inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                            residual_on=hydra_cfg.residual_on,
                        )
                else:
                    network = ResidualFlow_DiffEmb(
                        emb_nn=hydra_cfg.emb_nn,
                        return_flow_component=hydra_cfg.return_flow_component,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                    )
            else:
                network = ResidualFlow(
                    emb_nn=hydra_cfg.emb_nn,
                    return_flow_component=hydra_cfg.return_flow_component,
                    center_feature=hydra_cfg.center_feature,
                )
        elif hydra_cfg.flow_compute_type == 1:
            network = ResidualFlow_V1(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 2:
            network = ResidualFlow_V2(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 3:
            network = ResidualFlow_V3(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 4:
            network = ResidualFlow_V4(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 5:
            network = ResidualFlow_Correspondence(emb_nn=hydra_cfg.emb_nn)
        elif hydra_cfg.flow_compute_type == 6:
            network = ResidualFlow_Identity(emb_nn=hydra_cfg.emb_nn)
        elif hydra_cfg.flow_compute_type == "pe":
            network = ResidualFlow_PE(emb_nn=hydra_cfg.emb_nn)
        place_model_refinement = EquivarianceTestingModule(
            network,
            lr=hydra_cfg.lr,
            image_log_period=hydra_cfg.image_logging_period,
            weight_normalize=hydra_cfg.weight_normalize_place,
            loop=hydra_cfg.loop,
        )

        place_model_refinement.cuda()

        place_model_refinement.load_state_dict(
            torch.load(hydra_cfg.checkpoint_file_place_refinement)["state_dict"]
        )
        log_info(
            "Place Refinement Model Loaded from "
            + str(hydra_cfg.checkpoint_file_place_refinement)
        )

    if hydra_cfg.flow_compute_type == 0:
        if hydra_cfg.diff_emb:
            if hydra_cfg.diff_transformer:
                if hydra_cfg.mlp:
                    network = CorrespondenceFlow_DiffEmbMLP(
                        emb_dims=hydra_cfg.emb_dims,
                        emb_nn=hydra_cfg.emb_nn,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                    )
                else:
                    network = ResidualFlow_DiffEmbTransformer(
                        pred_weight=hydra_cfg.pred_weight,
                        emb_nn=hydra_cfg.emb_nn,
                        return_flow_component=hydra_cfg.return_flow_component,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                        residual_on=hydra_cfg.residual_on,
                    )
            else:
                network = ResidualFlow_DiffEmb(
                    emb_nn=hydra_cfg.emb_nn,
                    return_flow_component=hydra_cfg.return_flow_component,
                    center_feature=hydra_cfg.center_feature,
                    inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                )
        else:
            network = ResidualFlow(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
                center_feature=hydra_cfg.center_feature,
            )
    elif hydra_cfg.flow_compute_type == 1:
        network = ResidualFlow_V1(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 2:
        network = ResidualFlow_V2(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 3:
        network = ResidualFlow_V3(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 4:
        network = ResidualFlow_V4(
            emb_nn=hydra_cfg.emb_nn,
            return_flow_component=hydra_cfg.return_flow_component,
        )
    elif hydra_cfg.flow_compute_type == 5:
        network = ResidualFlow_Correspondence(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == 6:
        network = ResidualFlow_Identity(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == "pe":
        network = ResidualFlow_PE(emb_nn=hydra_cfg.emb_nn)
    grasp_model = EquivarianceTestingModule(
        network,
        lr=hydra_cfg.lr,
        image_log_period=hydra_cfg.image_logging_period,
        weight_normalize=hydra_cfg.weight_normalize_grasp,
        softmax_temperature=hydra_cfg.softmax_temperature_grasp,
        loop=hydra_cfg.loop,
    )

    grasp_model.cuda()

    if hydra_cfg.checkpoint_file_grasp is not None:
        grasp_model.load_state_dict(
            torch.load(hydra_cfg.checkpoint_file_grasp)["state_dict"]
        )
        log_info("Model Loaded from " + str(hydra_cfg.checkpoint_file_grasp))
    if hydra_cfg.checkpoint_file_grasp_refinement is not None:
        if hydra_cfg.flow_compute_type == 0:
            if hydra_cfg.diff_emb:
                if hydra_cfg.diff_transformer:
                    if hydra_cfg.mlp:
                        network = CorrespondenceFlow_DiffEmbMLP(
                            emb_dims=hydra_cfg.emb_dims,
                            emb_nn=hydra_cfg.emb_nn,
                            center_feature=hydra_cfg.center_feature,
                            inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                        )
                    else:
                        network = ResidualFlow_DiffEmbTransformer(
                            pred_weight=hydra_cfg.pred_weight,
                            emb_nn=hydra_cfg.emb_nn,
                            return_flow_component=hydra_cfg.return_flow_component,
                            center_feature=hydra_cfg.center_feature,
                            inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                            residual_on=hydra_cfg.residual_on,
                        )
                else:
                    network = ResidualFlow_DiffEmb(
                        emb_nn=hydra_cfg.emb_nn,
                        return_flow_component=hydra_cfg.return_flow_component,
                        center_feature=hydra_cfg.center_feature,
                        inital_sampling_ratio=hydra_cfg.inital_sampling_ratio,
                    )
            else:
                network = ResidualFlow(
                    emb_nn=hydra_cfg.emb_nn,
                    return_flow_component=hydra_cfg.return_flow_component,
                    center_feature=hydra_cfg.center_feature,
                )
        elif hydra_cfg.flow_compute_type == 1:
            network = ResidualFlow_V1(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 2:
            network = ResidualFlow_V2(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 3:
            network = ResidualFlow_V3(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 4:
            network = ResidualFlow_V4(
                emb_nn=hydra_cfg.emb_nn,
                return_flow_component=hydra_cfg.return_flow_component,
            )
        elif hydra_cfg.flow_compute_type == 5:
            network = ResidualFlow_Correspondence(emb_nn=hydra_cfg.emb_nn)
        elif hydra_cfg.flow_compute_type == 6:
            network = ResidualFlow_Identity(emb_nn=hydra_cfg.emb_nn)
        elif hydra_cfg.flow_compute_type == "pe":
            network = ResidualFlow_PE(emb_nn=hydra_cfg.emb_nn)
        grasp_model_refinement = EquivarianceTestingModule(
            network,
            lr=hydra_cfg.lr,
            image_log_period=hydra_cfg.image_logging_period,
            weight_normalize=hydra_cfg.weight_normalize_grasp,
            softmax_temperature=hydra_cfg.softmax_temperature_grasp,
            loop=hydra_cfg.loop,
        )

        grasp_model_refinement.cuda()

        if hydra_cfg.checkpoint_file_grasp_refinement is not None:
            grasp_model_refinement.load_state_dict(
                torch.load(hydra_cfg.checkpoint_file_grasp_refinement)["state_dict"]
            )
            log_info(
                "Model Grasp Refinement Model Loaded from "
                + str(hydra_cfg.checkpoint_file_grasp_refinement)
            )

    for iteration in range(hydra_cfg.start_iteration, hydra_cfg.num_iterations):
        # load a test object
        obj_shapenet_id = random.sample(test_object_ids, 1)[0]
        id_str = "Shapenet ID: %s" % obj_shapenet_id
        log_info(id_str)

        viz_dict = {}  # will hold information that's useful for post-run visualizations
        eval_iter_dir = osp.join(eval_save_dir, "trial_%d" % iteration)
        util.safe_makedirs(eval_iter_dir)

        if obj_class in ["bottle", "jar", "bowl", "mug"]:
            upright_orientation = common.euler2quat([np.pi / 2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(
            shapenet_obj_dir, obj_shapenet_id, "models/model_normalized.obj"
        )
        obj_obj_file_dec = obj_obj_file.split(".obj")[0] + "_dec.obj"

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if hydra_cfg.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        if hydra_cfg.any_pose:
            if obj_class in ["bowl", "bottle"]:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z,
            ]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi
            )
            pose_w_yaw = util.transform_pose(
                util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T)
            )
            pos, ori = (
                util.pose_stamped2list(pose_w_yaw)[:3],
                util.pose_stamped2list(pose_w_yaw)[3:],
            )
        else:
            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z,
            ]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi
            )
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = (
                util.pose_stamped2list(pose_w_yaw)[:3],
                util.pose_stamped2list(pose_w_yaw)[3:],
            )

        viz_dict["shapenet_id"] = obj_shapenet_id
        viz_dict["obj_obj_file"] = obj_obj_file
        if "normalized" not in shapenet_obj_dir:
            viz_dict["obj_obj_norm_file"] = osp.join(
                shapenet_obj_dir + "_normalized",
                obj_shapenet_id,
                "models/model_normalized.obj",
            )
        else:
            viz_dict["obj_obj_norm_file"] = osp.join(
                shapenet_obj_dir, obj_shapenet_id, "models/model_normalized.obj"
            )
        viz_dict["obj_obj_file_dec"] = obj_obj_file_dec
        viz_dict["mesh_scale"] = mesh_scale

        # convert mesh with vhacd
        if not osp.exists(obj_obj_file_dec):
            p.vhacd(
                obj_obj_file,
                obj_obj_file_dec,
                "log.txt",
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1,
            )

        robot.arm.go_home(ignore_physics=True)
        robot.arm.move_ee_xyz([0, 0, 0.2])

        if hydra_cfg.any_pose:
            robot.pb_client.set_step_sim(True)
        if obj_class in ["bowl"]:
            robot.pb_client.set_step_sim(True)

        obj_id = robot.pb_client.load_geom(
            "mesh",
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori,
        )
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)
        log_info("any_pose:{}".format(hydra_cfg.any_pose))
        if obj_class == "bowl":
            safeCollisionFilterPair(
                bodyUniqueIdA=obj_id,
                bodyUniqueIdB=table_id,
                linkIndexA=-1,
                linkIndexB=rack_link_id,
                enableCollision=False,
            )
            safeCollisionFilterPair(
                bodyUniqueIdA=obj_id,
                bodyUniqueIdB=table_id,
                linkIndexA=-1,
                linkIndexB=shelf_link_id,
                enableCollision=False,
            )
            robot.pb_client.set_step_sim(False)

        o_cid = None
        if hydra_cfg.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        hide_link(table_id, rack_link_id)

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1])
        )
        viz_dict["start_obj_pose"] = util.pose_stamped2list(obj_pose_world)

        if obj_class == "mug":
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        time.sleep(1.5)
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(eval_teleport_imgs_dir, "%d_init.png" % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        points_mug_raw, points_rack_raw = load_data_raw(
            num_points=1024,
            clouds=obj_points,
            classes=obj_classes,
            action_class=0,
            anchor_class=1,
        )
        if points_mug_raw is None:
            continue
        points_gripper_raw, points_mug_raw = load_data_raw(
            num_points=1024,
            clouds=obj_points,
            classes=obj_classes,
            action_class=2,
            anchor_class=0,
        )
        points_mug, points_rack = load_data(
            num_points=1024,
            clouds=obj_points,
            classes=obj_classes,
            action_class=0,
            anchor_class=1,
        )

        ans = place_model.get_transform(points_mug, points_rack)  # 1, 4, 4

        if hydra_cfg.checkpoint_file_place_refinement is not None:
            pred_points_action = ans["pred_points_action"]
            pred_T_action = ans["pred_T_action"]
            (
                points_trans_action,
                points_trans_anchor,
                points_action_mean,
            ) = place_model.action_centered(pred_points_action, points_rack)
            T_trans = pure_translation_se3(
                1, points_action_mean.squeeze(), device=points_trans_action.device
            )
            ans_refinement = place_model_refinement.get_transform(
                points_trans_action, points_trans_anchor
            )
            pred_T_action = pred_T_action.compose(
                T_trans.inverse()
                .compose(ans_refinement["pred_T_action"])
                .compose(T_trans)
            )
            ans_refinement["pred_T_action"] = pred_T_action
            ans = ans_refinement

        pred_T_action_init = ans["pred_T_action"]
        pred_T_action_mat = pred_T_action_init.get_matrix()[0].T.detach().cpu().numpy()
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)  # list
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1])
        )  # stamped_pose
        obj_start_pose = obj_pose_world
        rack_relative_pose = get_world_transform(
            pred_T_action_mat, obj_start_pose, points_mug_raw
        )  # pose_stamped
        obj_end_pose_list = util.pose_stamped2list(rack_relative_pose)
        transform_rack_relative_pose = util.get_transform(
            rack_relative_pose, obj_start_pose
        )
        pose_tuple = robot.arm.get_ee_pose()
        ee_pose_world = util.list2pose_stamped(
            list(pose_tuple[0]) + list(pose_tuple[1])
        )
        # Get Grasp Pose
        points_gripper, points_mug = load_data(
            num_points=1024,
            clouds=obj_points,
            classes=obj_classes,
            action_class=2,
            anchor_class=0,
        )
        ans_grasp = grasp_model.get_transform(points_gripper, points_mug)  # 1, 4, 4
        pred_T_action_init_gripper2mug = ans_grasp["pred_T_action"]
        pred_T_action_mat_gripper2mug = (
            pred_T_action_init_gripper2mug.get_matrix()[0].T.detach().cpu().numpy()
        )
        pred_T_action_mat_gripper2mug[2, -1] -= 0.001

        gripper_relative_pose = get_world_transform(
            pred_T_action_mat_gripper2mug, ee_pose_world, points_gripper_raw
        )  # transform from gripper to mug in world frame
        pre_grasp_ee_pose = util.pose_stamped2list(gripper_relative_pose)

        np.savez(
            f"{save_dir}/{iteration}_init_all_points.npz",
            clouds=cloud_points,
            colors=cloud_colors,
            classes=cloud_classes,
            shapenet_id=obj_shapenet_id,
        )

        np.savez(
            f"{save_dir}/{iteration}_init_obj_points.npz",
            clouds=obj_points,
            colors=obj_colors,
            classes=obj_classes,
            shapenet_id=obj_shapenet_id,
            points_mug_raw=points_mug_raw.detach().cpu(),
            points_gripper_raw=points_gripper_raw.detach().cpu(),
            points_rack_raw=points_rack_raw.detach().cpu(),
            pred_T_action_mat=pred_T_action_mat,
            pred_T_action_mat_gripper2mug=pred_T_action_mat_gripper2mug,
        )
        log_info("Saved point cloud data to:")
        log_info(f"{save_dir}/{iteration}_init_obj_points.npz")

        # optimize grasp pose
        viz_dict["start_ee_pose"] = pre_grasp_ee_pose

        ########################### grasp post-process #############################

        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        # pre_pre_grasp_ee_pose = pre_grasp_ee_pose
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf),
            )
        )

        # reset object to placement pose to detect placement success
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(
            obj_id, table_id, -1, placement_link_id, enableCollision=False
        )
        robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        np.savez(
            f"{save_dir}/{iteration}_teleport_all_points.npz",
            clouds=cloud_points,
            colors=cloud_colors,
            classes=cloud_classes,
            shapenet_id=obj_shapenet_id,
        )

        np.savez(
            f"{save_dir}/{iteration}_teleport_obj_points.npz",
            clouds=obj_points,
            colors=obj_colors,
            classes=obj_classes,
            shapenet_id=obj_shapenet_id,
        )

        time.sleep(1.0)
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, "teleport_%d.png" % iteration
        )
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(
            obj_id, table_id, -1, placement_link_id, enableCollision=True
        )
        robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        np.savez(
            f"{save_dir}/{iteration}_post_teleport_all_points.npz",
            clouds=cloud_points,
            colors=cloud_colors,
            classes=cloud_classes,
            shapenet_id=obj_shapenet_id,
        )

        np.savez(
            f"{save_dir}/{iteration}_post_teleport_obj_points.npz",
            clouds=obj_points,
            colors=obj_colors,
            classes=obj_classes,
            shapenet_id=obj_shapenet_id,
        )

        time.sleep(1.0)
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, "post_teleport_%d.png" % iteration
        )
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)

        obj_surf_contacts = p.getContactPoints(obj_id, table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        place_success_teleport_list.append(place_success_teleport)
        if not place_success_teleport:
            place_fail_teleport_list.append(iteration)

        time.sleep(1.0)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):
            # reset everything
            robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
            if hydra_cfg.any_pose:
                robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if hydra_cfg.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                robot.pb_client.set_step_sim(False)
            robot.arm.go_home(ignore_physics=True)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id,
                    bodyUniqueIdB=table_id,
                    linkIndexA=i,
                    linkIndexB=-1,
                    enableCollision=False,
                    physicsClientId=robot.pb_client.get_client_id(),
                )
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id,
                    bodyUniqueIdB=obj_id,
                    linkIndexA=i,
                    linkIndexB=-1,
                    enableCollision=False,
                    physicsClientId=robot.pb_client.get_client_id(),
                )
            robot.arm.eetool.open()

            if jnt_pos is None or grasp_jnt_pos is None:
                jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = ik_helper.get_feasible_ik(pre_grasp_ee_pose)

                if jnt_pos is None or grasp_jnt_pos is None:
                    jnt_pos = ik_helper.get_ik(pre_pre_grasp_ee_pose)
                    grasp_jnt_pos = ik_helper.get_ik(pre_grasp_ee_pose)

                    if jnt_pos is None or grasp_jnt_pos is None:
                        jnt_pos = robot.arm.compute_ik(
                            pre_pre_grasp_ee_pose[:3], pre_pre_grasp_ee_pose[3:]
                        )
                        # this is the pose that's at the grasp, where we just need to close the fingers
                        grasp_jnt_pos = robot.arm.compute_ik(
                            pre_grasp_ee_pose[:3], pre_grasp_ee_pose[3:]
                        )

            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(
                        eval_grasp_imgs_dir, "pre_grasp_%d.png" % iteration
                    )
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
                    obj_points, obj_colors, obj_classes = get_object_clouds(cams)

                    np.savez(
                        f"{save_dir}/{iteration}_pre_grasp_all_points.npz",
                        clouds=cloud_points,
                        colors=cloud_colors,
                        classes=cloud_classes,
                        shapenet_id=obj_shapenet_id,
                    )

                    np.savez(
                        f"{save_dir}/{iteration}_pre_grasp_obj_points.npz",
                        clouds=obj_points,
                        colors=obj_colors,
                        classes=obj_classes,
                        shapenet_id=obj_shapenet_id,
                    )

                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), jnt_pos)
                    plan2 = ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)

                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        robot.arm.eetool.open()
                        for jnt in plan1:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.025)
                        robot.arm.set_jpos(plan1[-1], wait=True)
                        for jnt in plan2:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.04)
                        robot.arm.set_jpos(grasp_plan[-1], wait=True)

                        # get pose that's straight up
                        offset_pose = util.transform_pose(
                            pose_source=util.list2pose_stamped(
                                np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
                            ),
                            pose_transform=util.list2pose_stamped(
                                [0, 0, 0.15, 0, 0, 0, 1]
                            ),
                        )
                        offset_pose_list = util.pose_stamped2list(offset_pose)
                        offset_jnts = ik_helper.get_feasible_ik(offset_pose_list)

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(
                                bodyUniqueIdA=robot.arm.robot_id,
                                bodyUniqueIdB=obj_id,
                                linkIndexA=i,
                                linkIndexB=-1,
                                enableCollision=True,
                                physicsClientId=robot.pb_client.get_client_id(),
                            )
                            safeCollisionFilterPair(
                                bodyUniqueIdA=robot.arm.robot_id,
                                bodyUniqueIdB=table_id,
                                linkIndexA=i,
                                linkIndexB=rack_link_id,
                                enableCollision=False,
                                physicsClientId=robot.pb_client.get_client_id(),
                            )

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[
                            0
                        ]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False
                        )
                        time.sleep(0.8)
                        grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                        grasp_img_fname = osp.join(
                            eval_grasp_imgs_dir, "post_grasp_%d.png" % iteration
                        )
                        np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
                        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

                        np.savez(
                            f"{save_dir}/{iteration}_post_grasp_all_points.npz",
                            clouds=cloud_points,
                            colors=cloud_colors,
                            classes=cloud_classes,
                            shapenet_id=obj_shapenet_id,
                        )

                        np.savez(
                            f"{save_dir}/{iteration}_post_grasp_obj_points.npz",
                            clouds=obj_points,
                            colors=obj_colors,
                            classes=obj_classes,
                            shapenet_id=obj_shapenet_id,
                        )

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(
                                robot, obj_id, right_pad_id, left_pad_id
                            )

                            if grasp_success:
                                # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                                safeCollisionFilterPair(
                                    obj_id, table_id, -1, -1, enableCollision=True
                                )
                                robot.arm.eetool.open()
                                p.resetBasePositionAndOrientation(
                                    obj_id, obj_pos_before_grasp, ori
                                )
                                soft_grasp_close(robot, finger_joint_id, force=40)
                                robot.arm.set_jpos(
                                    jnt_pos_before_grasp, ignore_physics=True
                                )
                                cid = constraint_grasp_close(robot, obj_id)
                                # grasp_rgb = robot.cam.get_images(get_rgb=True)[
                                #     0]
                                # grasp_img_fname = osp.join(
                                #     eval_grasp_imgs_dir, 'after_grasp_success_%d.png' % iteration)
                                # np2img(grasp_rgb.astype(
                                #     np.uint8), grasp_img_fname)
                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = ik_helper.plan_joint_motion(
                                robot.arm.get_jpos(), offset_jnts
                            )

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    robot.arm.set_jpos(jnt, wait=False)
                                    time.sleep(0.04)
                                robot.arm.set_jpos(offset_plan[-1], wait=True)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False
                        )
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, rack_link_id, enableCollision=False
                        )
                        time.sleep(1.0)

        if grasp_success:
            # optimize placement pose
            ee_end_pose = util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=transform_rack_relative_pose,
            )
            pre_ee_end_pose2 = util.transform_pose(
                pose_source=ee_end_pose, pose_transform=preplace_offset_tf
            )
            pre_ee_end_pose1 = util.transform_pose(
                pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf
            )

            ee_end_pose_list = util.pose_stamped2list(ee_end_pose)
            pre_ee_end_pose1_list = util.pose_stamped2list(pre_ee_end_pose1)
            pre_ee_end_pose2_list = util.pose_stamped2list(pre_ee_end_pose2)

            ####################################### get place pose ###########################################

            pre_place_jnt_pos1 = ik_helper.get_feasible_ik(pre_ee_end_pose1_list)
            pre_place_jnt_pos2 = ik_helper.get_feasible_ik(pre_ee_end_pose2_list)
            place_jnt_pos = ik_helper.get_feasible_ik(ee_end_pose_list)

            if (
                place_jnt_pos is not None
                and pre_place_jnt_pos2 is not None
                and pre_place_jnt_pos1 is not None
            ):
                plan1 = ik_helper.plan_joint_motion(
                    robot.arm.get_jpos(), pre_place_jnt_pos1
                )
                plan2 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos1, pre_place_jnt_pos2
                )
                plan3 = ik_helper.plan_joint_motion(pre_place_jnt_pos2, place_jnt_pos)

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.035)
                    robot.arm.set_jpos(place_plan[-1], wait=True)

                    ################################################################################################################

                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=True
                    )
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, rack_link_id, enableCollision=True
                    )

                    for jnt in plan3:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.075)
                    robot.arm.set_jpos(plan3[-1], wait=True)

                    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    robot.arm.eetool.open()

                    time.sleep(0.2)
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        safeCollisionFilterPair(
                            bodyUniqueIdA=robot.arm.robot_id,
                            bodyUniqueIdB=obj_id,
                            linkIndexA=i,
                            linkIndexB=-1,
                            enableCollision=False,
                            physicsClientId=robot.pb_client.get_client_id(),
                        )
                    robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=False
                    )
                    time.sleep(4.0)

                    # observe and record outcome
                    obj_surf_contacts = p.getContactPoints(
                        obj_id, table_id, -1, placement_link_id
                    )
                    touching_surf = len(obj_surf_contacts) > 0
                    obj_floor_contacts = p.getContactPoints(
                        obj_id, robot.arm.floor_id, -1, -1
                    )
                    touching_floor = len(obj_floor_contacts) > 0
                    place_success = touching_surf and not touching_floor

        robot.arm.go_home(ignore_physics=True)

        place_success_list.append(place_success)
        grasp_success_list.append(grasp_success)
        if not place_success:
            place_fail_list.append(iteration)
        if not grasp_success:
            grasp_fail_list.append(iteration)
        log_str = "Iteration: %d, " % iteration
        kvs = {}
        kvs["Place Success Rate"] = sum(place_success_list) / float(
            len(place_success_list)
        )
        kvs["Place [teleport] Success Rate"] = sum(place_success_teleport_list) / float(
            len(place_success_teleport_list)
        )
        kvs["Grasp Success Rate"] = sum(grasp_success_list) / float(
            len(grasp_success_list)
        )
        kvs["Place Success"] = place_success_list[-1]
        kvs["Place [teleport] Success"] = place_success_teleport_list[-1]
        kvs["Grasp Success"] = grasp_success_list[-1]

        overall_success_num = 0
        for i in range(len(grasp_success_list)):
            if place_success_teleport_list[i] == 1 and grasp_success_list[i] == 1:
                overall_success_num += 1
        kvs["overall success Rate"] = overall_success_num / float(
            len(grasp_success_list)
        )

        for k, v in kvs.items():
            log_str += "%s: %.3f, " % (k, v)
        id_str = ", shapenet_id: %s" % obj_shapenet_id
        log_info(log_str + id_str)
        f = open(txt_file_name, "a")
        f.write("{} \n".format("place success"))
        f.write(str(bool(place_success_list[-1])))
        f.write("\n")
        f.write("{} \n".format("Place [teleport] Success"))
        f.write(str(bool(place_success_teleport_list[-1])))
        f.write("\n")
        f.write("{} \n".format("Grasp Success"))
        f.write(str(bool(grasp_success_list[-1])))
        f.write("\n")
        f.close()

        eval_iter_dir = osp.join(eval_save_dir, "trial_%d" % iteration)
        if not osp.exists(eval_iter_dir):
            os.makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, "success_rate_eval_implicit.npz")
        np.savez(
            sample_fname,
            obj_shapenet_id=obj_shapenet_id,
            success=success_list,
            grasp_success=grasp_success,
            place_success=place_success,
            place_success_teleport=place_success_teleport,
            grasp_success_list=grasp_success_list,
            place_success_list=place_success_list,
            place_success_teleport_list=place_success_teleport_list,
            start_obj_pose=util.pose_stamped2list(obj_start_pose),
            best_place_obj_pose=obj_end_pose_list,
            mesh_file=obj_obj_file,
            distractor_info=None,
            args=hydra_cfg.__dict__,
            global_dict=global_dict,
            cfg=util.cn2dict(cfg),
            obj_cfg=util.cn2dict(obj_cfg),
        )

        robot.pb_client.remove_body(obj_id)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, util.signal_handler)

    main()
