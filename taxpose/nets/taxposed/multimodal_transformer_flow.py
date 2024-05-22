#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import copy
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.taxposed.pointnet2 import PointNet2MSG, PointNet2SSG
from taxpose.nets.taxposed.pointnet2pyg import (
    PN2DenseParams,
    PN2DenseWrapper,
    PN2EncoderParams,
    PN2EncoderWrapper,
    PN2HybridWrapper,
)
from taxpose.nets.vn_dgcnn import VN_DGCNN, VNArgs
from taxpose.utils.conditioning_utils import gumbel_softmax_topk
from taxpose.utils.sample_utils import sample_closest_pairs, sample_random_pair

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


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
    def __init__(
        self,
        emb_dims=512,
        input_dims=3,
        num_heads=1,
        conditioning_size=0,
        last_relu=True,
    ):
        super(DGCNN, self).__init__()
        self.num_heads = num_heads
        self.conditioning_size = conditioning_size
        self.last_relu = last_relu

        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        if self.num_heads == 1:
            self.conv5 = nn.Conv2d(
                512 + self.conditioning_size, emb_dims, kernel_size=1, bias=False
            )
            self.bn5 = nn.BatchNorm2d(emb_dims)
        else:
            if self.conditioning_size > 0:
                raise NotImplementedError(
                    "Conditioning not implemented for multi-head DGCNN"
                )
            self.conv5s = nn.ModuleList(
                [
                    nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
                    for _ in range(self.num_heads)
                ]
            )
            self.bn5s = nn.ModuleList(
                [nn.BatchNorm2d(emb_dims) for _ in range(self.num_heads)]
            )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        # bn5 defined above

    def forward(self, x, conditioning=None):
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

        if self.conditioning_size == 0:
            assert conditioning is None
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            assert conditioning is not None
            x = torch.cat((x1, x2, x3, x4, conditioning[:, :, :, None]), dim=1)
            # x = torch.cat((x1, x2, x3, x4, torch.tile(conditioning[:,:,None,:], (1, 1, num_points, 1))), dim=1)

        if self.num_heads == 1:
            x = self.bn5(self.conv5(x)).view(batch_size, -1, num_points)
        else:
            x = [
                bn5(conv5(x)).view(batch_size, -1, num_points)
                for bn5, conv5 in zip(self.bn5s, self.conv5s)
            ]

        if self.last_relu:
            if self.num_heads == 1:
                x = F.relu(x)
            else:
                x = [F.relu(head) for head in x]
        return x


class DGCNNClassification(nn.Module):
    # Reference: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py#L88-L153

    def __init__(
        self, emb_dims=512, input_dims=3, num_heads=1, dropout=0.5, output_channels=40
    ):
        super(DGCNNClassification, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        if self.num_heads == 1:
            self.linear3 = nn.Linear(256, self.output_channels)
        else:
            self.linear3s = nn.ModuleList(
                [nn.Linear(256, self.output_channels) for _ in range(self.num_heads)]
            )

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

        x = self.conv5(x).squeeze()
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        if self.num_heads == 1:
            x = self.linear3(x)[:, :, None]
        else:
            x = [linear3(x)[:, :, None] for linear3 in self.linear3s]

        return x


class DGCNNHybrid(nn.Module):
    """
    DGCNN with both classification and segmentation heads
    """

    def __init__(
        self,
        emb_dims=512,
        input_dims=3,
        num_heads=1,
        conditioning_size=0,
        last_relu=True,
        dropout=0.5,
        output_channels=40,
    ):
        super(DGCNNHybrid, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.conditioning_size = conditioning_size
        self.last_relu = last_relu
        self.dropout = dropout
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(self.input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(
            512 + self.conditioning_size, emb_dims, kernel_size=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        if self.num_heads == 1:
            self.linear3 = nn.Linear(256, self.output_channels)
        else:
            self.linear3s = nn.ModuleList(
                [nn.Linear(256, self.output_channels) for _ in range(self.num_heads)]
            )

    def forward(self, x, conditioning=None):
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

        if self.conditioning_size == 0:
            assert conditioning is None
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            assert conditioning is not None
            x = torch.cat((x1, x2, x3, x4, conditioning[:, :, :, None]), dim=1)
            # x = torch.cat((x1, x2, x3, x4, torch.tile(conditioning[:,:,None,:], (1, 1, num_points, 1))), dim=1)

        x = self.conv5(x)

        class_x = x.squeeze()
        seg_x = self.bn5(x).view(batch_size, -1, num_points)

        if self.last_relu:
            seg_x = F.relu(x)

        class_x1 = F.adaptive_max_pool1d(class_x, 1).view(batch_size, -1)
        class_x2 = F.adaptive_avg_pool1d(class_x, 1).view(batch_size, -1)
        class_x = torch.cat((class_x1, class_x2), 1)

        class_x = F.leaky_relu(self.bn6(self.linear1(class_x)), negative_slope=0.2)
        class_x = self.dp1(class_x)
        class_x = F.leaky_relu(self.bn7(self.linear2(class_x)), negative_slope=0.2)
        class_x = self.dp2(class_x)

        if self.num_heads == 1:
            class_x = self.linear3(class_x)[:, :, None]
        else:
            class_x = [linear3(class_x)[:, :, None] for linear3 in self.linear3s]

        return seg_x, class_x


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


class Multimodal_ResidualFlow_DiffEmbTransformer(nn.Module):
    EMB_DIMS_BY_CONDITIONING = {
        "pos_delta_l2norm": 1,
        "pos_exp_delta_l2norm": 1,
        "uniform_prior_pos_delta_l2norm": 1,
        "distance_prior_pos_delta_l2norm": 1,
        "pos_delta_l2norm_dist_vec": 4,
        "uniform_prior_pos_delta_l2norm_dist_vec": 4,
        "distance_prior_pos_delta_l2norm_dist_vec": 4,
        # 'latent_z': 1, # Make the dimensions as close as possible to the ablations we're comparing this against
        # 'latent_z_1pred': 1, # Same
        # 'latent_z_1pred_10d': 10, # Same
        "hybrid_pos_delta_l2norm": 1,
        "hybrid_pos_delta_l2norm_internalcond": 1,
        "hybrid_pos_delta_l2norm_global": 1,
        "hybrid_pos_delta_l2norm_global_internalcond": 1,
        "latent_z_linear": 512,
        "latent_z_linear_internalcond": 512,
        "pos_delta_vec": 1,
        "pos_onehot": 1,
        "pos_loc3d": 3,
    }

    # Number of heads that the DGCNN should output
    NUM_HEADS_BY_CONDITIONING = {
        "pos_delta_l2norm": 1,
        "pos_exp_delta_l2norm": 1,
        "uniform_prior_pos_delta_l2norm": 1,
        "distance_prior_pos_delta_l2norm": 1,
        "pos_delta_l2norm_dist_vec": 1,
        "uniform_prior_pos_delta_l2norm_dist_vec": 1,
        "distance_prior_pos_delta_l2norm_dist_vec": 1,
        # 'latent_z': 2, # One for mu and one for var
        # 'latent_z_1pred': 2, # Same
        # 'latent_z_1pred_10d': 2, # Same
        "hybrid_pos_delta_l2norm": 1,
        "hybrid_pos_delta_l2norm_internalcond": 1,
        "hybrid_pos_delta_l2norm_global": 2,
        "hybrid_pos_delta_l2norm_global_internalcond": 2,
        "latent_z_linear": 2,
        "latent_z_linear_internalcond": 2,
        "pos_delta_vec": 1,
        "pos_onehot": 1,
        "pos_loc3d": 1,
    }

    DEPRECATED_CONDITIONINGS = ["latent_z", "latent_z_1pred", "latent_z_1pred_10d"]

    TP_INPUT_DIMS = {
        "pos_delta_l2norm": 3 + 1,
        "pos_exp_delta_l2norm": 3 + 1,
        "uniform_prior_pos_delta_l2norm": 3 + 1,
        "distance_prior_pos_delta_l2norm": 3 + 1,
        "pos_delta_l2norm_dist_vec": 3 + 1,
        "uniform_prior_pos_delta_l2norm_dist_vec": 3 + 1,
        "distance_prior_pos_delta_l2norm_dist_vec": 3 + 1,
        # Not implemented because it's dynamic. Also this isn't used anymore
        "hybrid_pos_delta_l2norm": 3
        + 1,  # Add cfg.latent_z_linear_size to this in main script because its dynamic, hacky
        "hybrid_pos_delta_l2norm_internalcond": 3
        + 1,  # Add cfg.latent_z_linear_size to this in main script because its dynamic, hacky
        "hybrid_pos_delta_l2norm_global": 3
        + 1,  # Add cfg.latent_z_linear_size to this in main script because its dynamic, hacky
        "hybrid_pos_delta_l2norm_global_internalcond": 3
        + 1,  # Add cfg.latent_z_linear_size to this in main script because its dynamic, hacky
        "latent_z_linear": 3,  # Add cfg.latent_z_linear_size to this in main script because its dynamic, hacky
        "latent_z_linear_internalcond": 3,
        "pos_delta_vec": 3 + 3,
        "pos_onehot": 3 + 1,
        "pos_loc3d": 3 + 3,
        "latent_3d_z": 3 + 3,
    }

    def __init__(
        self,
        residualflow_diffembtransformer,
        gumbel_temp=0.5,
        freeze_residual_flow=False,
        center_feature=False,
        freeze_z_embnn=False,
        division_smooth_factor=1,
        add_smooth_factor=0.05,
        conditioning="pos_delta_l2norm",
        latent_z_linear_size=40,
        taxpose_centering="mean",
        use_action_z=True,
        pzY_encoder_type="dgcnn",
        pzY_transformer="none",
        pzY_transformer_embnn_dims=512,
        pzY_transformer_emb_dims=512,
        pzY_input_dims=3,
        pzY_dropout_goal_emb=0.0,
        pzY_embedding_routine="joint",
        pzY_embedding_option=0,
        hybrid_cond_logvar_limit=0.0,
        latent_z_cond_logvar_limit=0.0,
        closest_point_conditioning=None,
    ):
        super(Multimodal_ResidualFlow_DiffEmbTransformer, self).__init__()

        assert taxpose_centering in ["mean", "z"]
        assert (
            conditioning not in self.DEPRECATED_CONDITIONINGS
        ), f"This conditioning {conditioning} is deprecated and should not be used"
        assert conditioning in self.EMB_DIMS_BY_CONDITIONING.keys()

        self.latent_z_linear_size = latent_z_linear_size
        self.conditioning = conditioning
        self.taxpose_centering = taxpose_centering
        # if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
        #     assert not freeze_residual_flow and not freeze_z_embnn, "Prob didn't want to freeze residual flow or z embnn when using latent_z_linear"
        self.tax_pose = residualflow_diffembtransformer
        self.freeze_residual_flow = freeze_residual_flow
        self.center_feature = center_feature
        self.freeze_z_embnn = freeze_z_embnn
        self.freeze_embnn = self.tax_pose.freeze_embnn
        self.gumbel_temp = gumbel_temp
        self.division_smooth_factor = division_smooth_factor
        self.add_smooth_factor = add_smooth_factor
        self.use_action_z = use_action_z
        self.dropout_goal_emb = pzY_dropout_goal_emb

        self.hybrid_cond_logvar_limit = hybrid_cond_logvar_limit
        self.latent_z_cond_logvar_limit = latent_z_cond_logvar_limit

        # Embedding networks
        self.input_dims = pzY_input_dims
        self.emb_dims = self.EMB_DIMS_BY_CONDITIONING[self.conditioning]

        if self.conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
        ]:
            self.emb_dims += self.latent_z_linear_size

        self.num_emb_heads = self.NUM_HEADS_BY_CONDITIONING[self.conditioning]
        self.pzY_encoder_type = pzY_encoder_type
        self.closest_point_conditioning = closest_point_conditioning

        # Point cloud with class labels between action and anchor
        if self.conditioning not in [
            "latent_z_linear",
            "latent_z_linear_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            # Single encoder
            if self.pzY_encoder_type == "dgcnn":
                print(f"--- P(z|Y) Using 1 DGCNN ---")
                self.emb_nn_objs_at_goal = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.pzY_encoder_type == "vn_dgcnn":
                assert self.input_dims == 3, "Only support 3D points for VN_DGCNN"
                print(f"--- P(z|Y) Using 1 VN_DGCNN ---")
                args = VNArgs()
                self.emb_nn_objs_at_goal = VN_DGCNN(
                    args, num_part=self.emb_dims, gc=False
                )
            elif self.pzY_encoder_type == "pn++_msg":
                print(f"--- P(z|Y) Using 1 PN++ MSG ---")
                self.emb_nn_objs_at_goal = PointNet2MSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.pzY_encoder_type == "pn++_ssg":
                print(f"--- P(z|Y) Using 1 PN++ SSG ---")
                self.emb_nn_objs_at_goal = PointNet2SSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.pzY_encoder_type == "pn++":
                print(f"--- P(z|Y) Using 1 PyG PN++ ---")
                args = PN2DenseParams()
                self.emb_nn_objs_at_goal = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )

            # Two encoders
            elif self.pzY_encoder_type == "2_dgcnn":
                print(f"--- P(z|Y) Using 2 DGCNN ---")
                self.emb_nn_action_at_goal = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
                self.emb_nn_anchor_at_goal = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.pzY_encoder_type == "2_pn++_msg":
                print(f"--- P(z|Y) Using 2 PN++ MSG ---")
                self.emb_nn_action_at_goal = PointNet2MSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
                self.emb_nn_anchor_at_goal = PointNet2MSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.pzY_encoder_type == "2_pn++_ssg":
                print(f"--- P(z|Y) Using 2 PN++ SSG ---")
                self.emb_nn_action_at_goal = PointNet2SSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
                self.emb_nn_anchor_at_goal = PointNet2SSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.pzY_encoder_type == "2_pn++":
                print(f"--- P(z|Y) Using 2 PyG PN++ ---")
                args = PN2DenseParams()
                self.emb_nn_action_at_goal = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )
                self.emb_nn_anchor_at_goal = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )
            else:
                raise ValueError(
                    f"pzY_encoder_type {self.pzY_encoder_type} not implemented for conditioning {self.conditioning}"
                )
        elif self.conditioning in [
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            if self.pzY_encoder_type == "dgcnn":
                print(f"--- P(z|Y) Using 1 DGCNN Hybrid ---")
                self.emb_nn_objs_at_goal = DGCNNHybrid(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                    output_channels=self.latent_z_linear_size,
                )
            elif self.pzY_encoder_type == "pn++":
                print(f"--- P(z|Y) Using 1 PyG PN++ Hybrid ---")
                args = PN2DenseParams()
                self.emb_nn_objs_at_goal = PN2HybridWrapper(
                    in_channels=self.input_dims - 3,
                    out_channels=self.latent_z_linear_size,
                    num_heads=self.num_emb_heads,
                    emb_dims=self.emb_dims,
                    p=args,
                )
        else:
            if self.pzY_encoder_type == "dgcnn":
                print(f"--- P(z|Y) Using 1 DGCNN Classification ---")
                self.emb_nn_objs_at_goal = DGCNNClassification(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    dropout=0.5,
                    output_channels=self.latent_z_linear_size,
                )
            elif self.pzY_encoder_type == "pn++":
                print(f"--- P(z|Y) Using 1 PyG PN++ Classification ---")
                args = PN2EncoderParams()
                self.emb_nn_objs_at_goal = PN2EncoderWrapper(
                    in_channels=self.input_dims - 3,
                    out_channels=self.latent_z_linear_size,
                    num_heads=self.num_emb_heads,
                    emb_dims=self.emb_dims,
                    p=args,
                )
            else:
                raise ValueError(
                    f"pzY_encoder_type {self.pzY_encoder_type} not implemented for conditioning {self.conditioning}"
                )

        # Set up transformer
        self.pzY_transformer = pzY_transformer
        self.pzY_transformer_embnn_dims = pzY_transformer_embnn_dims
        self.pzY_transformer_emb_dims = pzY_transformer_emb_dims
        if self.pzY_transformer != "none":
            if self.pzY_transformer == "cross_object":
                print(f"--- P(z|Y) Using Cross Object Transformer ---")
                if self.pzY_encoder_type == "2_dgcnn":
                    print(f"------ With 2 DGCNN Encoders ------")
                    self.emb_nn_action_at_goal = DGCNN(
                        input_dims=self.input_dims,
                        emb_dims=self.pzY_transformer_embnn_dims,
                        num_heads=self.num_emb_heads,
                        last_relu=False,
                    )
                    self.emb_nn_anchor_at_goal = DGCNN(
                        input_dims=self.input_dims,
                        emb_dims=self.pzY_transformer_embnn_dims,
                        num_heads=self.num_emb_heads,
                        last_relu=False,
                    )
                elif self.pzY_encoder_type == "2_pn++_msg":
                    print(f"------ With 2 PN++ MSG Encoders ------")
                    self.emb_nn_action_at_goal = PointNet2MSG(
                        num_classes=self.pzY_transformer_embnn_dims,
                        additional_channel=self.input_dims - 3,
                    )
                    self.emb_nn_anchor_at_goal = PointNet2MSG(
                        num_classes=self.pzY_transformer_embnn_dims,
                        additional_channel=self.input_dims - 3,
                    )
                elif self.pzY_encoder_type == "2_pn++_ssg":
                    print(f"------ With 2 PN++ SSG Encoders ------")
                    self.emb_nn_action_at_goal = PointNet2SSG(
                        num_classes=self.pzY_transformer_embnn_dims,
                        additional_channel=self.input_dims - 3,
                    )
                    self.emb_nn_anchor_at_goal = PointNet2SSG(
                        num_classes=self.pzY_transformer_embnn_dims,
                        additional_channel=self.input_dims - 3,
                    )
                elif self.pzY_encoder_type == "2_pn++":
                    print(f"------ With 2 PyG PN++ Encoders ------")
                    args = PN2DenseParams()
                    self.emb_nn_action_at_goal = PN2DenseWrapper(
                        in_channels=self.input_dims - 3,
                        out_channels=self.pzY_transformer_embnn_dims,
                        p=args,
                    )
                    self.emb_nn_anchor_at_goal = PN2DenseWrapper(
                        in_channels=self.input_dims - 3,
                        out_channels=self.pzY_transformer_embnn_dims,
                        p=args,
                    )
                else:
                    raise ValueError(
                        f"pzY_transformer {self.pzY_transformer} not implemented for encoder_type {self.pzY_encoder_type}"
                    )

                self.action_transformer = Transformer(
                    emb_dims=self.pzY_transformer_emb_dims,
                    return_attn=True,
                    bidirectional=False,
                )
                self.anchor_transformer = Transformer(
                    emb_dims=self.pzY_transformer_emb_dims,
                    return_attn=True,
                    bidirectional=False,
                )

                self.action_proj = nn.Sequential(
                    PointNet([self.pzY_transformer_emb_dims, 64, 64, 64, 128, 512]),
                    nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                )
                self.anchor_proj = nn.Sequential(
                    PointNet([self.pzY_transformer_emb_dims, 64, 64, 64, 128, 512]),
                    nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                )
            else:
                raise ValueError(f"Unknown pzY_transformer {self.pzY_transformer}.")

        # Auxiliary networks
        self.embedding_routine = pzY_embedding_routine
        self.embedding_option = pzY_embedding_option
        if self.embedding_routine in ["joint2global"]:
            assert self.conditioning in [
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ], "joint2global only supported for hybrid global conditioning"

            self.cond_embnn_type = self.pzY_encoder_type
            self.cond_emb_dims = self.emb_dims

            if self.cond_embnn_type == "pn++":
                args = PN2EncoderParams()
                self.cond_embnn = PN2EncoderWrapper(
                    in_channels=self.emb_dims,
                    out_channels=self.latent_z_linear_size,
                    num_heads=self.num_emb_heads,
                    emb_dims=self.emb_dims,
                    p=args,
                )

                args = PN2DenseParams()
                self.emb_nn_objs_at_goal = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )

    def get_dense_translation_point(self, points, ref, conditioning):
        """
        points- point cloud. (B, 3, num_points)
        ref- one hot vector (or nearly one-hot) that denotes the reference point
                 (B, num_points)

        Returns:
            dense point cloud. Each point contains the distance to the reference point (B, 3 or 1, num_points)
        """
        assert ref.ndim == 2
        assert torch.allclose(
            ref.sum(axis=1),
            torch.full((ref.shape[0], 1), 1, dtype=torch.float, device=ref.device),
        )
        num_points = points.shape[2]
        reference = (points * ref[:, None, :]).sum(axis=2)
        if conditioning in [
            "pos_delta_l2norm",
            "uniform_prior_pos_delta_l2norm",
            "distance_prior_pos_delta_l2norm",
            "pos_delta_l2norm_dist_vec",
            "uniform_prior_pos_delta_l2norm_dist_vec",
            "distance_prior_pos_delta_l2norm_dist_vec",
        ]:
            dense = torch.norm(reference[:, :, None] - points, dim=1, keepdim=True)
        elif conditioning == "pos_exp_delta_l2norm":
            dense = torch.exp(
                -torch.norm(reference[:, :, None] - points, dim=1, keepdim=True) / 1
            )
        elif conditioning == "pos_delta_vec":
            dense = reference[:, :, None] - points
        elif conditioning == "pos_loc3d":
            dense = reference[:, :, None].repeat(1, 1, 1024)
        elif conditioning == "pos_onehot":
            dense = ref[:, None, :]
        else:
            raise ValueError(
                f"Conditioning {conditioning} probably doesn't require a dense representation. This function is for"
                + "['pos_delta_l2norm', 'pos_delta_vec', 'pos_loc3d', 'pos_onehot', 'uniform_prior_pos_delta_l2norm']"
            )
        #        dense = reference[:, :, None] - points
        #        dense = torch.abs(reference[:, :, None] - points)
        #        dense = torch.exp(-dense / 0.1)
        #        dense = torch.exp(-dense.abs() / 0.1)
        return dense, reference

    def sample_dense_embedding(self, goal_emb, sampling_method="gumbel", n_samples=1):
        """Sample the dense goal embedding"""

        # Disable dropout if this is a validation forward pass
        d = self.dropout_goal_emb if self.training else 0
        mask = torch.rand(goal_emb.shape) > d
        goal_emb[mask == False] = float("-inf")

        samples = []
        if sampling_method == "gumbel" and n_samples > 1:
            khot, onehot_list = gumbel_softmax_topk(
                goal_emb, k=n_samples, tau=self.gumbel_temp, hard=True, dim=-1
            )
            samples = onehot_list
        else:
            for i in range(n_samples):
                if sampling_method == "gumbel":
                    sample = F.gumbel_softmax(
                        goal_emb, self.gumbel_temp, hard=True, dim=-1
                    )

                elif sampling_method == "random":
                    # Note that this does not interact with self.dropout_goal_emb
                    rand_idx = torch.randint(
                        0, goal_emb.shape[-1], (goal_emb.shape[0],)
                    )
                    sample = (
                        torch.nn.functional.one_hot(
                            rand_idx, num_classes=goal_emb.shape[-1]
                        )
                        .float()
                        .to(goal_emb.device)
                    )

                elif sampling_method == "top_n":
                    top_idxs = torch.topk(F.softmax(goal_emb), n_samples, dim=-1)[1]
                    sample = (
                        torch.nn.functional.one_hot(
                            top_idxs[:, i], num_classes=goal_emb.shape[-1]
                        )
                        .float()
                        .to(goal_emb.device)
                    )

                else:
                    raise ValueError(
                        f"Sampling method {sampling_method} not implemented"
                    )

                samples.append(sample)
        return samples

    def add_single_conditioning(
        self,
        goal_emb,
        points,
        conditioning,
        cond_type="action",
        sampling_method="gumbel",
        n_samples=1,
        z_samples=None,
    ):
        for_debug = {}

        sample_outputs = []
        if conditioning in [
            "pos_delta_vec",
            "pos_loc3d",
            "pos_onehot",
            "pos_delta_l2norm",
            "pos_exp_delta_l2norm",
            "uniform_prior_pos_delta_l2norm",
            "distance_prior_pos_delta_l2norm",
            "pos_delta_l2norm_dist_vec",
            "uniform_prior_pos_delta_l2norm_dist_vec",
            "distance_prior_pos_delta_l2norm_dist_vec",
        ]:

            goal_emb = (goal_emb + self.add_smooth_factor) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:, 0, :]

            translation_samples = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            if z_samples is not None:
                translation_samples = z_samples[f"translation_samples_{cond_type}"]

            for translation_sample in translation_samples:
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt, ref = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None, points, translation_sample, conditioning=conditioning
                    )
                )

                if not self.use_action_z and cond_type == "action":
                    # Replace with 0's of the same shape:
                    dense_trans_pt = torch.zeros_like(dense_trans_pt)

                points_and_cond = torch.cat([points] + [dense_trans_pt], axis=1)

                for_debug = {
                    f"dense_trans_pt_{cond_type}": dense_trans_pt,
                    f"trans_pt_{cond_type}": ref,
                    f"trans_sample_{cond_type}": translation_sample,
                    f"{cond_type}_points_and_cond": points_and_cond,
                }

                sample_outputs.append(
                    {
                        f"{cond_type}_points_and_cond": points_and_cond,
                        "for_debug": for_debug,
                    }
                )
        elif conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            assert (
                self.latent_z_linear_size % 2 == 0
            ), "latent_z_linear_size must be even for hybrid conditioning"

            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu

            # Do dense part first
            goal_emb_dense = goal_emb[:, :1]
            goal_emb_dense = (
                goal_emb_dense + self.add_smooth_factor
            ) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_dense_translation = goal_emb_dense[:, 0, :]

            # Sample the spatial conditioning point(s)
            translation_samples = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_dense_translation,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            # Optionally override the samples with pre-sampled ones
            if z_samples is not None:
                translation_samples = z_samples[f"translation_samples_{cond_type}"]

            # For each sampled point, get the conditioning
            for translation_sample in translation_samples:
                # Turn spatial conditioning point into dense representation
                dense_trans_pt, ref = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        points,
                        translation_sample,
                        conditioning="pos_delta_l2norm",
                    )
                )

                for_debug = {}
                # Also get the selected points latent z from the per-point latents
                if conditioning in [
                    "hybrid_pos_delta_l2norm",
                    "hybrid_pos_delta_l2norm_internalcond",
                ]:
                    # Extract the mu and logvar from the dense embeddings
                    goal_emb_mu = goal_emb[:, 1 : self.latent_z_linear_size // 2 + 1]
                    goal_emb_logvar = goal_emb[:, self.latent_z_linear_size // 2 + 1 :]

                    if (
                        self.hybrid_cond_logvar_limit > 0
                    ):  # Hacky way to prevent large logvar causing NaNs
                        goal_emb_logvar = self.hybrid_cond_logvar_limit * torch.tanh(
                            goal_emb_logvar
                        )

                    # Get the latent z corresponding to the selected action/anchor points
                    selected_mu = (goal_emb_mu * translation_sample[:, None, :]).sum(
                        axis=2, keepdim=True
                    )
                    selected_logvar = (
                        goal_emb_logvar * translation_sample[:, None, :]
                    ).sum(axis=2, keepdim=True)
                    selected_latent = reparametrize(selected_mu, selected_logvar)

                    for_debug = {
                        **for_debug,
                        f"goal_emb_mu_{cond_type}": goal_emb_mu,
                        f"goal_emb_logvar_{cond_type}": goal_emb_logvar,
                        f"{cond_type}_mu": selected_mu,
                        f"{cond_type}_logvar": selected_logvar,
                        f"{cond_type}_latent": selected_latent,
                    }
                # Also get a global latent z
                elif conditioning in [
                    "hybrid_pos_delta_l2norm_global",
                    "hybrid_pos_delta_l2norm_global_internalcond",
                ]:
                    # Extract the mu and logvar from global embeddings
                    goal_emb_mu = goal_emb[1]
                    goal_emb_logvar = goal_emb[2]
                    goal_emb = goal_emb[0]

                    if (
                        self.hybrid_cond_logvar_limit > 0
                    ):  # Hacky way to prevent large logvar causing NaNs
                        goal_emb_logvar = self.hybrid_cond_logvar_limit * torch.tanh(
                            goal_emb_logvar
                        )

                    selected_latent = reparametrize(goal_emb_mu, goal_emb_logvar)

                    for_debug = {
                        **for_debug,
                        "goal_emb_mu": goal_emb_mu,
                        "goal_emb_logvar": goal_emb_logvar,
                    }
                else:
                    raise ValueError(
                        f"Hybrid conditioning {conditioning} does not exist."
                    )

                # Add the dense conditioning to the points
                points_and_cond = torch.cat(
                    [
                        points,
                        dense_trans_pt,
                    ],
                    axis=1,
                )

                # Add the latent z to the points
                if conditioning in [
                    "hybrid_pos_delta_l2norm",
                    "hybrid_pos_delta_l2norm_global",
                ]:
                    points_and_cond = torch.cat(
                        [
                            points_and_cond,
                            torch.tile(
                                selected_latent, (1, 1, selected_latent.shape[-1])
                            ),
                        ],
                        axis=1,
                    )

                for_debug = {
                    **for_debug,
                    f"dense_trans_pt_{cond_type}": dense_trans_pt,
                    f"trans_pt_{cond_type}": ref,
                    f"trans_sample_{cond_type}": translation_sample,
                    f"goal_emb_latent": selected_latent,
                }

                sample_outputs.append(
                    {
                        f"{cond_type}_points_and_cond": points_and_cond,
                        "for_debug": for_debug,
                    }
                )

        else:
            raise ValueError(
                f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}"
            )

        return sample_outputs

    # TODO: rename to add_joint_conditioning, or merge the two functions
    def add_conditioning(
        self,
        goal_emb,
        action_points,
        anchor_points,
        conditioning,
        sampling_method="gumbel",
        n_samples=1,
        z_samples=None,
    ):
        for_debug = {}

        sample_outputs = []
        if conditioning in [
            "pos_delta_vec",
            "pos_loc3d",
            "pos_onehot",
            "pos_delta_l2norm",
            "pos_exp_delta_l2norm",
            "uniform_prior_pos_delta_l2norm",
            "distance_prior_pos_delta_l2norm",
            "pos_delta_l2norm_dist_vec",
            "uniform_prior_pos_delta_l2norm_dist_vec",
            "distance_prior_pos_delta_l2norm_dist_vec",
        ]:

            goal_emb = (goal_emb + self.add_smooth_factor) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:, 0, :]

            goal_emb_translation_action = goal_emb_translation[
                :, : action_points.shape[2]
            ]
            goal_emb_translation_anchor = goal_emb_translation[
                :, action_points.shape[2] :
            ]

            translation_samples_action = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_action,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )
            translation_samples_anchor = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_anchor,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            if z_samples is not None:
                translation_samples_action = z_samples["translation_samples_action"]
                translation_samples_anchor = z_samples["translation_samples_anchor"]

            for translation_sample_action, translation_sample_anchor in zip(
                translation_samples_action, translation_samples_anchor
            ):
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt_action, ref_action = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        action_points,
                        translation_sample_action,
                        conditioning=conditioning,
                    )
                )
                dense_trans_pt_anchor, ref_anchor = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        anchor_points,
                        translation_sample_anchor,
                        conditioning=conditioning,
                    )
                )

                if not self.use_action_z:
                    # Replace with 0's of the same shape:
                    dense_trans_pt_action = torch.zeros_like(dense_trans_pt_action)

                action_points_and_cond = torch.cat(
                    [action_points] + [dense_trans_pt_action], axis=1
                )
                anchor_points_and_cond = torch.cat(
                    [anchor_points] + [dense_trans_pt_anchor], axis=1
                )

                for_debug = {
                    "dense_trans_pt_action": dense_trans_pt_action,
                    "dense_trans_pt_anchor": dense_trans_pt_anchor,
                    "trans_pt_action": ref_action,
                    "trans_pt_anchor": ref_anchor,
                    "trans_sample_action": translation_sample_action,
                    "trans_sample_anchor": translation_sample_anchor,
                    "action_points_and_cond": action_points_and_cond,
                    "anchor_points_and_cond": anchor_points_and_cond,
                }

                sample_outputs.append(
                    {
                        "action_points_and_cond": action_points_and_cond,
                        "anchor_points_and_cond": anchor_points_and_cond,
                        "for_debug": for_debug,
                    }
                )
        elif conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Do the reparametrization trick on the predicted mu and var
            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu

            # Here, the goal emb has 2 heads. One for mean and one for variance
            goal_emb_mu = goal_emb[0]
            goal_emb_logvar = goal_emb[1]

            if (
                self.latent_z_cond_logvar_limit > 0
            ):  # Hacky way to prevent large logvar causing NaNs
                goal_emb_logvar = self.latent_z_cond_logvar_limit * torch.tanh(
                    goal_emb_logvar
                )

            for i in range(n_samples):
                goal_emb_sample = reparametrize(goal_emb_mu, goal_emb_logvar)

                for_debug = {
                    "goal_emb_mu": goal_emb_mu,
                    "goal_emb_logvar": goal_emb_logvar,
                    "goal_emb_sample": goal_emb_sample,
                }

                if conditioning == "latent_z_linear":
                    action_points_and_cond = torch.cat(
                        [action_points]
                        + [
                            torch.tile(goal_emb_sample, (1, 1, action_points.shape[-1]))
                        ],
                        axis=1,
                    )
                    anchor_points_and_cond = torch.cat(
                        [anchor_points]
                        + [
                            torch.tile(goal_emb_sample, (1, 1, anchor_points.shape[-1]))
                        ],
                        axis=1,
                    )
                elif conditioning == "latent_z_linear_internalcond":
                    # The cond will be added in by TAXPose
                    action_points_and_cond = action_points
                    anchor_points_and_cond = anchor_points
                else:
                    raise ValueError("Why is it here?")

                sample_outputs.append(
                    {
                        "action_points_and_cond": action_points_and_cond,
                        "anchor_points_and_cond": anchor_points_and_cond,
                        "for_debug": for_debug,
                    }
                )
        elif conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            assert (
                self.latent_z_linear_size % 2 == 0
            ), "latent_z_linear_size must be even for hybrid conditioning"

            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu

            if conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
            ]:
                goal_emb_mu = goal_emb[:, 1 : self.latent_z_linear_size // 2 + 1]
                goal_emb_mu_action = goal_emb_mu[:, :, : action_points.shape[2]]
                goal_emb_mu_anchor = goal_emb_mu[:, :, action_points.shape[2] :]

                goal_emb_logvar = goal_emb[:, self.latent_z_linear_size // 2 + 1 :]
                goal_emb_logvar_action = goal_emb_logvar[:, :, : action_points.shape[2]]
                goal_emb_logvar_anchor = goal_emb_logvar[:, :, action_points.shape[2] :]

                if (
                    self.hybrid_cond_logvar_limit > 0
                ):  # Hacky way to prevent large logvar causing NaNs
                    goal_emb_logvar_action = self.hybrid_cond_logvar_limit * torch.tanh(
                        goal_emb_logvar_action
                    )
                    goal_emb_logvar_anchor = self.hybrid_cond_logvar_limit * torch.tanh(
                        goal_emb_logvar_anchor
                    )

            else:
                goal_emb_mu = goal_emb[1]
                goal_emb_logvar = goal_emb[2]
                goal_emb = goal_emb[0]

                if (
                    self.hybrid_cond_logvar_limit > 0
                ):  # Hacky way to prevent large logvar causing NaNs
                    goal_emb_logvar = self.hybrid_cond_logvar_limit * torch.tanh(
                        goal_emb_logvar
                    )

            # Do dense part first
            goal_emb_dense = goal_emb[:, :1]
            goal_emb_dense = (
                goal_emb_dense + self.add_smooth_factor
            ) / self.division_smooth_factor

            goal_emb_translation = goal_emb_dense[:, 0, :]
            goal_emb_translation_action = goal_emb_translation[
                :, : action_points.shape[2]
            ]
            goal_emb_translation_anchor = goal_emb_translation[
                :, action_points.shape[2] :
            ]

            translation_samples_action = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_action,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )
            translation_samples_anchor = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_anchor,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            if z_samples is not None:
                raise ValueError("z_samples not implemented for hybrid conditioning")

            for translation_sample_action, translation_sample_anchor in zip(
                translation_samples_action, translation_samples_anchor
            ):
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt_action, ref_action = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        action_points,
                        translation_sample_action,
                        conditioning="pos_delta_l2norm",
                    )
                )
                dense_trans_pt_anchor, ref_anchor = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        anchor_points,
                        translation_sample_anchor,
                        conditioning="pos_delta_l2norm",
                    )
                )

                for_debug = {}
                if self.conditioning in [
                    "hybrid_pos_delta_l2norm",
                    "hybrid_pos_delta_l2norm_internalcond",
                ]:
                    # Get the latent z corresponding to the selected action/anchor points
                    action_mu = (
                        goal_emb_mu_action * translation_sample_action[:, None, :]
                    ).sum(axis=2, keepdim=True)
                    action_logvar = (
                        goal_emb_logvar_action * translation_sample_action[:, None, :]
                    ).sum(axis=2, keepdim=True)
                    action_latent = reparametrize(action_mu, action_logvar)

                    anchor_mu = (
                        goal_emb_mu_anchor * translation_sample_anchor[:, None, :]
                    ).sum(axis=2, keepdim=True)
                    anchor_logvar = (
                        goal_emb_logvar_anchor * translation_sample_anchor[:, None, :]
                    ).sum(axis=2, keepdim=True)
                    anchor_latent = reparametrize(anchor_mu, anchor_logvar)

                    # Concat into the final goal_emb_latent
                    goal_emb_latent = torch.cat([action_latent, anchor_latent], axis=1)

                    for_debug = {
                        **for_debug,
                        "goal_emb_mu_action": goal_emb_mu_action,
                        "goal_emb_mu_anchor": goal_emb_mu_anchor,
                        "goal_emb_logvar_action": goal_emb_logvar_action,
                        "goal_emb_logvar_anchor": goal_emb_logvar_anchor,
                        "action_mu": action_mu,
                        "action_logvar": action_logvar,
                        "action_latent": action_latent,
                        "anchor_mu": anchor_mu,
                        "anchor_logvar": anchor_logvar,
                        "anchor_latent": anchor_latent,
                    }
                else:
                    goal_emb_latent = reparametrize(goal_emb_mu, goal_emb_logvar)

                    for_debug = {
                        **for_debug,
                        "goal_emb_mu": goal_emb_mu,
                        "goal_emb_logvar": goal_emb_logvar,
                    }

                action_points_and_cond = torch.cat(
                    [
                        action_points,
                        dense_trans_pt_action,
                    ],
                    axis=1,
                )
                anchor_points_and_cond = torch.cat(
                    [
                        anchor_points,
                        dense_trans_pt_anchor,
                    ],
                    axis=1,
                )

                for_debug = {
                    **for_debug,
                    "dense_trans_pt_action": dense_trans_pt_action,
                    "dense_trans_pt_anchor": dense_trans_pt_anchor,
                    "trans_pt_action": ref_action,
                    "trans_pt_anchor": ref_anchor,
                    "trans_sample_action": translation_sample_action,
                    "trans_sample_anchor": translation_sample_anchor,
                    "goal_emb_latent": goal_emb_latent,
                }

                sample_outputs.append(
                    {
                        "action_points_and_cond": action_points_and_cond,
                        "anchor_points_and_cond": anchor_points_and_cond,
                        "for_debug": for_debug,
                    }
                )

        else:
            raise ValueError(
                f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}"
            )

        return sample_outputs

    def forward(
        self,
        *input,
        mode="forward",
        sampling_method="gumbel",
        n_samples=1,
        z_samples=None,
    ):
        # Forward pass goes through all of the model
        # Inference will use a sample from the prior if there is one
        #     - ex: conditioning = latent_z_linear_internalcond
        assert mode in ["forward", "inference"]

        action_points = input[0].permute(0, 2, 1)[
            :, : self.input_dims
        ]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, : self.input_dims]

        if input[2] is None:
            mode = "inference"

        embedding_samples = []
        if mode == "forward":
            # Get the demonstration point clouds
            goal_action_points = input[2].permute(0, 2, 1)[:, : self.input_dims]
            goal_anchor_points = input[3].permute(0, 2, 1)[:, : self.input_dims]

            # Prepare the goal point clouds
            goal_action_points_dmean = goal_action_points
            goal_anchor_points_dmean = goal_anchor_points
            if self.center_feature:
                mean_goal = torch.cat(
                    [goal_action_points[:, :3], goal_anchor_points[:, :3]], axis=-1
                ).mean(dim=2, keepdim=True)
                goal_action_points_dmean = goal_action_points[:, :3] - mean_goal
                goal_anchor_points_dmean = goal_anchor_points[:, :3] - mean_goal

                goal_action_points_dmean = torch.cat(
                    [goal_action_points_dmean, goal_action_points[:, 3:]], axis=1
                )
                goal_anchor_points_dmean = torch.cat(
                    [goal_anchor_points_dmean, goal_anchor_points[:, 3:]], axis=1
                )

            # Get the action and anchor embeddings jointly
            if self.embedding_routine == "joint":
                # Concatenate the action and anchor points
                goal_points_dmean = torch.cat(
                    [goal_action_points_dmean, goal_anchor_points_dmean], axis=2
                )

                # Obtain a goal embedding
                with torch.set_grad_enabled(not self.freeze_z_embnn):
                    if self.pzY_encoder_type in [
                        "2_dgcnn",
                        "2_pn++_msg",
                        "2_pn++_ssg",
                        "2_pn++",
                    ]:
                        # Get separate action/anchor embeddings
                        action_emb = self.emb_nn_action_at_goal(
                            goal_action_points_dmean
                        )
                        anchor_emb = self.emb_nn_anchor_at_goal(
                            goal_anchor_points_dmean
                        )

                        # Apply cross-object transformer
                        if self.pzY_transformer in ["cross_object"]:
                            action_emb_tf, action_attn = (
                                self.emb_nn_action_at_goal.transformer(
                                    action_emb, anchor_emb
                                )
                            )
                            anchor_emb_tf, anchor_attn = (
                                self.emb_nn_anchor_at_goal.transformer(
                                    anchor_emb, action_emb
                                )
                            )

                            action_emb = self.action_proj(action_emb_tf)
                            anchor_emb = self.anchor_proj(anchor_emb_tf)
                        elif self.pzY_transformer in ["none"]:
                            pass
                        else:
                            raise ValueError(
                                f"pzY_transformer {self.pzY_transformer} not implemented for encoder_type {self.pzY_encoder_type}"
                            )

                        # Concatenate the action and anchor embeddings
                        goal_emb = torch.cat([action_emb, anchor_emb], dim=-1)
                    else:
                        # Get the joint action/anchor embeddings
                        goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean)

                # Intermediate processing
                if self.conditioning in [
                    "hybrid_pos_delta_l2norm_global",
                    "hybrid_pos_delta_l2norm_global_internalcond",
                ]:
                    goal_emb_seg, goal_emb_cls = goal_emb
                    goal_emb = [goal_emb_seg, *goal_emb_cls]

                additional_logging = {}
                if self.conditioning in [
                    "pos_delta_l2norm_dist_vec",
                    "uniform_prior_pos_delta_l2norm_dist_vec",
                    "distance_prior_pos_delta_l2norm_dist_vec",
                ]:
                    dist_vec = goal_emb[:, 1:]
                    goal_emb = goal_emb[:, :1]
                    additional_logging["dist_vec"] = dist_vec

                if self.closest_point_conditioning is not None:
                    match = re.match(
                        r"^top_([0-9]+)_(0\.[0-9]+)$", self.closest_point_conditioning
                    )
                    top_k = int(match.group(1))
                    prob = float(match.group(2))
                    if np.random.rand() < prob:
                        action_z_samples = []
                        anchor_z_samples = []
                        for i in range(n_samples):
                            (
                                top_k_action_indices,
                                top_k_anchor_indices,
                                top_k_distances,
                            ) = sample_closest_pairs(
                                action_points.permute(0, 2, 1),
                                anchor_points.permute(0, 2, 1),
                                top_k=top_k,
                            )
                            selected_action_indices, selected_anchor_indices, _ = (
                                sample_random_pair(
                                    top_k_action_indices,
                                    top_k_anchor_indices,
                                    top_k_distances,
                                )
                            )

                            selected_action_onehots = (
                                torch.nn.functional.one_hot(
                                    selected_action_indices.reshape(-1, 1),
                                    num_classes=action_points.shape[-1],
                                )
                                .float()
                                .squeeze(1)
                            )
                            selected_anchor_onehots = (
                                torch.nn.functional.one_hot(
                                    selected_anchor_indices.reshape(-1, 1),
                                    num_classes=anchor_points.shape[-1],
                                )
                                .float()
                                .squeeze(1)
                            )

                            action_z_samples.append(selected_action_onehots)
                            anchor_z_samples.append(selected_anchor_onehots)

                        z_samples = {
                            "translation_samples_action": action_z_samples,
                            "translation_samples_anchor": anchor_z_samples,
                        }

                # Return n samples of spatially conditioned action and anchor points
                embedding_samples = self.add_conditioning(
                    goal_emb,
                    action_points[:, :3],  # Use only XYZ
                    anchor_points[:, :3],  # Use only XYZ
                    self.conditioning,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                    z_samples=z_samples,
                )

                # Keep track of additional logging
                for i in range(len(embedding_samples)):
                    embedding_samples[i]["goal_emb"] = goal_emb
                    embedding_samples[i]["for_debug"] = {
                        **embedding_samples[i]["for_debug"],
                        **additional_logging,
                    }

            # Get the action and anchor embeddings, then get global latent z
            elif self.embedding_routine in ["joint2global"]:
                # Concatenate the action and anchor points
                goal_points_dmean = torch.cat(
                    [goal_action_points_dmean, goal_anchor_points_dmean], axis=2
                )

                # Obtain a goal embedding
                with torch.set_grad_enabled(not self.freeze_z_embnn):
                    if self.pzY_encoder_type in ["pn++"]:
                        goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean)

                        if self.closest_point_conditioning is not None:
                            match = re.match(
                                r"^top_([0-9]+)_(0\.[0-9]+)$",
                                self.closest_point_conditioning,
                            )
                            top_k = int(match.group(1))
                            prob = float(match.group(2))
                            if np.random.rand() < prob:
                                action_z_samples = []
                                anchor_z_samples = []
                                for i in range(n_samples):
                                    (
                                        top_k_action_indices,
                                        top_k_anchor_indices,
                                        top_k_distances,
                                    ) = sample_closest_pairs(
                                        action_points.permute(0, 2, 1),
                                        anchor_points.permute(0, 2, 1),
                                        top_k=top_k,
                                    )
                                    (
                                        selected_action_indices,
                                        selected_anchor_indices,
                                        _,
                                    ) = sample_random_pair(
                                        top_k_action_indices,
                                        top_k_anchor_indices,
                                        top_k_distances,
                                    )

                                    selected_action_onehots = (
                                        torch.nn.functional.one_hot(
                                            selected_action_indices.reshape(-1, 1),
                                            num_classes=action_points.shape[-1],
                                        )
                                        .float()
                                        .squeeze(1)
                                    )
                                    selected_anchor_onehots = (
                                        torch.nn.functional.one_hot(
                                            selected_anchor_indices.reshape(-1, 1),
                                            num_classes=anchor_points.shape[-1],
                                        )
                                        .float()
                                        .squeeze(1)
                                    )

                                    action_z_samples.append(selected_action_onehots)
                                    anchor_z_samples.append(selected_anchor_onehots)

                                z_samples = {
                                    "translation_samples_action": action_z_samples,
                                    "translation_samples_anchor": anchor_z_samples,
                                }

                        # Get initial action and anchor embeddings
                        embedding_samples = self.add_conditioning(
                            goal_emb,
                            action_points[:, :3],  # Use only XYZ
                            anchor_points[:, :3],  # Use only XYZ
                            conditioning="pos_delta_l2norm",
                            sampling_method=sampling_method,
                            n_samples=n_samples,
                            z_samples=z_samples,
                        )

                        def reparametrize(mu, logvar):
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            return eps * std + mu

                        for embedding_sample in embedding_samples:
                            action_points_and_cond = embedding_sample[
                                "action_points_and_cond"
                            ]
                            anchor_points_and_cond = embedding_sample[
                                "anchor_points_and_cond"
                            ]

                            # Center action and anchor XYZ
                            action_points_and_cond_xyz_centered = (
                                action_points_and_cond[:, :3]
                                - action_points_and_cond[:, :3].mean(
                                    dim=2, keepdim=True
                                )
                            )
                            anchor_points_and_cond_xyz_centered = (
                                anchor_points_and_cond[:, :3]
                                - anchor_points_and_cond[:, :3].mean(
                                    dim=2, keepdim=True
                                )
                            )

                            # Concatenate the centered XYZ with the rest of the features
                            action_points_and_cond = torch.cat(
                                [
                                    action_points_and_cond_xyz_centered,
                                    action_points_and_cond[:, 3:],
                                ],
                                axis=1,
                            )
                            anchor_points_and_cond = torch.cat(
                                [
                                    anchor_points_and_cond_xyz_centered,
                                    anchor_points_and_cond[:, 3:],
                                ],
                                axis=1,
                            )

                            cond_emb_points = torch.cat(
                                [action_points_and_cond, anchor_points_and_cond], axis=2
                            )
                            cond_emb = self.cond_embnn(cond_emb_points)

                            goal_emb_mu, goal_emb_logvar = cond_emb
                            goal_emb_latent = reparametrize(
                                goal_emb_mu, goal_emb_logvar
                            )

                            embedding_sample["goal_emb"] = [goal_emb, *cond_emb]
                            embedding_sample["for_debug"]["goal_emb_mu"] = goal_emb_mu
                            embedding_sample["for_debug"][
                                "goal_emb_logvar"
                            ] = goal_emb_logvar
                            embedding_sample["for_debug"][
                                "goal_emb_latent"
                            ] = goal_emb_latent

                    else:
                        raise ValueError(
                            f"pzY_encoder_type {self.pzY_encoder_type} not implemented for embedding_routine {self.embedding_routine}"
                        )

            # Get the action and anchor sequentially, action->anchor->goal emb. or anchor->action->goal emb.
            elif self.embedding_routine in ["action2anchor", "anchor2action"]:
                raise NotImplementedError("This is not implemented yet")
            else:
                raise ValueError(f"Unknown embedding_routine {self.embedding_routine}")

        elif mode == "inference":
            # Return n samples of spatially conditioned action and anchor points
            for i in range(n_samples):
                action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug = (
                    self.sample(action_points[:, :3], anchor_points[:, :3])
                )
                embedding_samples.append(
                    {
                        "action_points_and_cond": action_points_and_cond,
                        "anchor_points_and_cond": anchor_points_and_cond,
                        "goal_emb": goal_emb,
                        "for_debug": for_debug,
                    }
                )

        else:
            raise ValueError(f"Unknown mode {mode}")

        # Do the TAXPose forward pass
        outputs = []
        for embedding_sample in embedding_samples:
            action_points_and_cond = embedding_sample["action_points_and_cond"]
            anchor_points_and_cond = embedding_sample["anchor_points_and_cond"]
            goal_emb = embedding_sample["goal_emb"]
            for_debug = embedding_sample["for_debug"]

            # Optionally prepare the internal TAXPose DGCNN conditioning
            tax_pose_conditioning_action = None
            tax_pose_conditioning_anchor = None
            if self.conditioning == "latent_z_linear_internalcond":
                tax_pose_conditioning_action = torch.tile(
                    for_debug["goal_emb_sample"], (1, 1, action_points.shape[-1])
                )
                tax_pose_conditioning_anchor = torch.tile(
                    for_debug["goal_emb_sample"], (1, 1, anchor_points.shape[-1])
                )
            elif self.conditioning in [
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                tax_pose_conditioning_action = torch.tile(
                    for_debug["goal_emb_latent"], (1, 1, action_points.shape[-1])
                )
                tax_pose_conditioning_anchor = torch.tile(
                    for_debug["goal_emb_latent"], (1, 1, anchor_points.shape[-1])
                )

            # Prepare the action and anchor points for TAXPose
            if self.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_global",
            ]:
                action_points_and_cond = torch.cat(
                    [
                        action_points_and_cond,
                        torch.tile(
                            for_debug["goal_emb_latent"],
                            (1, 1, action_points.shape[-1]),
                        ),
                    ],
                    axis=1,
                )
                anchor_points_and_cond = torch.cat(
                    [
                        anchor_points_and_cond,
                        torch.tile(
                            for_debug["goal_emb_latent"],
                            (1, 1, anchor_points.shape[-1]),
                        ),
                    ],
                    axis=1,
                )

            if self.taxpose_centering == "mean":
                # Mean center the action and anchor points
                action_center = action_points[:, :3].mean(dim=2, keepdim=True)
                anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
            elif self.taxpose_centering == "z":
                # Center about the selected discrete spatial grounding action/anchor point
                action_center = for_debug["trans_pt_action"][:, :, None]
                anchor_center = for_debug["trans_pt_anchor"][:, :, None]
            else:
                raise ValueError(
                    f"Unknown self.taxpose_centering: {self.taxpose_centering}"
                )

            # Decode spatially conditioned action and anchor points into flows to obtain the goal configuration
            with torch.set_grad_enabled(not self.freeze_residual_flow):
                flow_action = self.tax_pose(
                    action_points_and_cond.permute(0, 2, 1),
                    anchor_points_and_cond.permute(0, 2, 1),
                    conditioning_action=tax_pose_conditioning_action,
                    conditioning_anchor=tax_pose_conditioning_anchor,
                    action_center=action_center,
                    anchor_center=anchor_center,
                )

            ########## LOGGING ############
            # Change goal_emb here to be what is going to be logged. For the latent_z conditioning, we just log the mean
            if (
                self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]
                and goal_emb is not None
            ):
                goal_emb = goal_emb[0]  # This is mu
            elif self.conditioning in [
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                goal_emb = goal_emb[0]  # This is point distribution

            flow_action = {
                **flow_action,
                "goal_emb": goal_emb,
                **for_debug,
            }

            outputs.append(flow_action)
        return outputs

    def sample_single(self, points, sample_type="action", **input_kwargs):
        if self.conditioning in [
            "uniform_prior_pos_delta_l2norm",
            "uniform_prior_pos_delta_l2norm_dist_vec",
        ]:
            # sample from a uniform prior
            N, B = points.shape[-1], points.shape[0]
            translation_sample = F.one_hot(torch.randint(N, (B,)), N).float().cuda()

            dense_trans_pt, ref = (
                Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                    None, points, translation_sample, conditioning=self.conditioning
                )
            )

            points_and_cond = torch.cat([points] + [dense_trans_pt], axis=1)

            goal_emb = None

            for_debug = {
                f"dense_trans_pt_{sample_type}": dense_trans_pt,
                f"trans_pt_{sample_type}": ref,
                f"trans_sample_{sample_type}": translation_sample,
            }
        elif self.conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            assert (
                "goal_emb" in input_kwargs
            ), "goal_emb must be passed in for hybrid conditioning"
            assert (
                "sampling_method" in input_kwargs
            ), "sampling_method must be passed in for hybrid conditioning"
            assert (
                "n_samples" in input_kwargs
            ), "n_samples must be passed in for hybrid conditioning"
            goal_emb = input_kwargs["goal_emb"]
            sampling_method = input_kwargs["sampling_method"]
            n_samples = input_kwargs["n_samples"]

            # Do dense part first
            goal_emb_dense = goal_emb[:, :1]
            goal_emb_dense = (
                goal_emb_dense + self.add_smooth_factor
            ) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_dense_translation = goal_emb_dense[:, 0, :]

            # Sample the spatial conditioning point(s)
            translation_samples = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_dense_translation,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            # For each sampled point, get the conditioning
            sample_outputs = []
            for translation_sample in translation_samples:
                # Turn spatial conditioning point into dense representation
                dense_trans_pt, ref = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        points,
                        translation_sample,
                        conditioning="pos_delta_l2norm",
                    )
                )

                latent = torch.randn(
                    (points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(points.device)

                points_and_cond = torch.cat(
                    [
                        points,
                        dense_trans_pt,
                    ],
                    axis=1,
                )

                # TODO: Remove this, hacky way to be compatible w/ model without cond. x compute_loss() during pzX compute_loss()
                goal_emb_mu = torch.zeros(
                    (points.shape[0], self.latent_z_linear_size // 2, points.shape[-1])
                ).to(points.device)
                goal_emb_logvar = torch.ones(
                    (points.shape[0], self.latent_z_linear_size // 2, points.shape[-1])
                ).to(points.device)

                mu = torch.randn(
                    (points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(points.device)
                logvar = torch.randn(
                    (points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(points.device)

                for_debug = {
                    f"dense_trans_pt_{sample_type}": dense_trans_pt,
                    f"trans_pt_{sample_type}": ref,
                    f"trans_sample_{sample_type}": translation_sample,
                    f"goal_emb_mu_{sample_type}": goal_emb_mu,
                    f"goal_emb_logvar_{sample_type}": goal_emb_logvar,
                    f"{sample_type}_mu": mu,
                    f"{sample_type}_logvar": logvar,
                    f"{sample_type}_latent": latent,
                }

                sample_outputs.append(
                    {
                        f"{sample_type}_points_and_cond": points_and_cond,
                        "goal_emb": goal_emb,
                        "for_debug": for_debug,
                    }
                )

            return sample_outputs
        else:
            raise ValueError(
                f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings"
            )
        return points_and_cond, goal_emb, for_debug

    def sample(self, action_points, anchor_points, **input_kwargs):
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Take a SINGLE sample z ~ N(0,1)
            for_debug = {}
            goal_emb_action = None
            goal_emb_anchor = None
            if self.conditioning == "latent_z_linear":
                goal_emb = torch.tile(
                    torch.randn(
                        (action_points.shape[0], self.latent_z_linear_size, 1)
                    ).to(action_points.device),
                    (1, 1, action_points.shape[-1]),
                )
                action_points_and_cond = torch.cat([action_points, goal_emb], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points, goal_emb], axis=1)
            elif self.conditioning == "latent_z_linear_internalcond":
                goal_emb_sample = torch.randn(
                    (action_points.shape[0], self.latent_z_linear_size, 1)
                ).to(action_points.device)
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug["goal_emb_sample"] = goal_emb_sample
                goal_emb = None
            else:
                raise ValueError("Why is it here?")
        elif self.conditioning in [
            "uniform_prior_pos_delta_l2norm",
            "uniform_prior_pos_delta_l2norm_dist_vec",
        ]:
            # sample from a uniform prior
            N_action, N_anchor, B = (
                action_points.shape[-1],
                anchor_points.shape[-1],
                action_points.shape[0],
            )
            translation_sample_action = (
                F.one_hot(torch.randint(N_action, (B,)), N_action).float().cuda()
            )
            translation_sample_anchor = (
                F.one_hot(torch.randint(N_anchor, (B,)), N_anchor).float().cuda()
            )

            dense_trans_pt_action, ref_action = (
                Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                    None,
                    action_points,
                    translation_sample_action,
                    conditioning=self.conditioning,
                )
            )
            dense_trans_pt_anchor, ref_anchor = (
                Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                    None,
                    anchor_points,
                    translation_sample_anchor,
                    conditioning=self.conditioning,
                )
            )

            action_points_and_cond = torch.cat(
                [action_points] + [dense_trans_pt_action], axis=1
            )
            anchor_points_and_cond = torch.cat(
                [anchor_points] + [dense_trans_pt_anchor], axis=1
            )

            goal_emb = None

            for_debug = {
                "dense_trans_pt_action": dense_trans_pt_action,
                "dense_trans_pt_anchor": dense_trans_pt_anchor,
                "trans_pt_action": ref_action,
                "trans_pt_anchor": ref_anchor,
                "trans_sample_action": translation_sample_action,
                "trans_sample_anchor": translation_sample_anchor,
            }
        elif self.conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:
            assert (
                "goal_emb" in input_kwargs
            ), "goal_emb must be passed in for hybrid conditioning"
            assert (
                "sampling_method" in input_kwargs
            ), "sampling_method must be passed in for hybrid conditioning"
            assert (
                "n_samples" in input_kwargs
            ), "n_samples must be passed in for hybrid conditioning"
            goal_emb = input_kwargs["goal_emb"]
            sampling_method = input_kwargs["sampling_method"]
            n_samples = input_kwargs["n_samples"]

            # Do dense part first
            goal_emb_dense = goal_emb[:, :1]
            goal_emb_dense = (
                goal_emb_dense + self.add_smooth_factor
            ) / self.division_smooth_factor

            goal_emb_translation = goal_emb_dense[:, 0, :]
            goal_emb_translation_action = goal_emb_translation[
                :, : action_points.shape[2]
            ]
            goal_emb_translation_anchor = goal_emb_translation[
                :, action_points.shape[2] :
            ]

            translation_samples_action = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_action,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )
            translation_samples_anchor = (
                Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                    self,
                    goal_emb_translation_anchor,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            )

            sample_outputs = []
            for translation_sample_action, translation_sample_anchor in zip(
                translation_samples_action, translation_samples_anchor
            ):
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt_action, ref_action = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        action_points,
                        translation_sample_action,
                        conditioning="pos_delta_l2norm",
                    )
                )
                dense_trans_pt_anchor, ref_anchor = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                        None,
                        anchor_points,
                        translation_sample_anchor,
                        conditioning="pos_delta_l2norm",
                    )
                )

                action_latent = torch.randn(
                    (action_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(action_points.device)
                anchor_latent = torch.randn(
                    (anchor_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(anchor_points.device)

                # Concat into the final goal_emb_latent
                goal_emb_latent = torch.cat([action_latent, anchor_latent], axis=1)

                action_points_and_cond = torch.cat(
                    [
                        action_points,
                        dense_trans_pt_action,
                    ],
                    axis=1,
                )
                anchor_points_and_cond = torch.cat(
                    [
                        anchor_points,
                        dense_trans_pt_anchor,
                    ],
                    axis=1,
                )

                # TODO: Remove this, hacky way to be compatible w/ model without cond. x compute_loss() during pzX compute_loss()
                goal_emb_mu_action = torch.zeros(
                    (
                        action_points.shape[0],
                        self.latent_z_linear_size // 2,
                        action_points.shape[-1],
                    )
                ).to(action_points.device)
                goal_emb_mu_anchor = torch.zeros(
                    (
                        anchor_points.shape[0],
                        self.latent_z_linear_size // 2,
                        anchor_points.shape[-1],
                    )
                ).to(anchor_points.device)
                goal_emb_logvar_action = torch.ones(
                    (
                        action_points.shape[0],
                        self.latent_z_linear_size // 2,
                        action_points.shape[-1],
                    )
                ).to(action_points.device)
                goal_emb_logvar_anchor = torch.ones(
                    (
                        anchor_points.shape[0],
                        self.latent_z_linear_size // 2,
                        anchor_points.shape[-1],
                    )
                ).to(anchor_points.device)

                action_mu = torch.randn(
                    (action_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(action_points.device)
                action_logvar = torch.randn(
                    (action_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(action_points.device)
                anchor_mu = torch.randn(
                    (anchor_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(anchor_points.device)
                anchor_logvar = torch.randn(
                    (anchor_points.shape[0], self.latent_z_linear_size // 2, 1)
                ).to(anchor_points.device)

                for_debug = {
                    "dense_trans_pt_action": dense_trans_pt_action,
                    "dense_trans_pt_anchor": dense_trans_pt_anchor,
                    "trans_pt_action": ref_action,
                    "trans_pt_anchor": ref_anchor,
                    "trans_sample_action": translation_sample_action,
                    "trans_sample_anchor": translation_sample_anchor,
                    "goal_emb_mu_action": goal_emb_mu_action,
                    "goal_emb_mu_anchor": goal_emb_mu_anchor,
                    "goal_emb_logvar_action": goal_emb_logvar_action,
                    "goal_emb_logvar_anchor": goal_emb_logvar_anchor,
                    "action_mu": action_mu,
                    "action_logvar": action_logvar,
                    "action_latent": action_latent,
                    "anchor_mu": anchor_mu,
                    "anchor_logvar": anchor_logvar,
                    "anchor_latent": anchor_latent,
                    "goal_emb_mu": torch.cat([action_mu, anchor_mu], axis=1),
                    "goal_emb_logvar": torch.cat(
                        [action_logvar, anchor_logvar], axis=1
                    ),
                    "goal_emb_latent": goal_emb_latent,
                }

                sample_outputs.append(
                    {
                        "action_points_and_cond": action_points_and_cond,
                        "anchor_points_and_cond": anchor_points_and_cond,
                        "goal_emb": goal_emb,
                        "for_debug": for_debug,
                    }
                )

            return sample_outputs
        else:
            raise ValueError(
                f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings"
            )
        return action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug


class Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(nn.Module):
    def __init__(
        self,
        residualflow_embnn,
        encoder_type="2_dgcnn",
        sample_z=True,
        shuffle_for_pzX=False,
        return_debug=False,
        use_action_z=True,
        pzX_transformer="none",
        pzX_transformer_embnn_dims=512,
        pzX_transformer_emb_dims=512,
        pzX_input_dims=3,
        pzX_dropout_goal_emb=0.0,
        hybrid_cond_pzX_sample_latent=False,
        pzX_embedding_routine="joint",
        pzX_embedding_option=0,
    ):
        super(Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX, self).__init__()
        self.residflow_embnn = residualflow_embnn

        # Use the other class definition so that it matches between classes
        self.conditioning = self.residflow_embnn.conditioning
        self.num_emb_heads = self.residflow_embnn.num_emb_heads
        self.emb_dims = self.residflow_embnn.emb_dims
        self.input_dims = pzX_input_dims
        self.taxpose_centering = self.residflow_embnn.taxpose_centering
        self.freeze_residual_flow = self.residflow_embnn.freeze_residual_flow
        self.freeze_z_embnn = self.residflow_embnn.freeze_z_embnn
        self.freeze_embnn = self.residflow_embnn.freeze_embnn

        self.shuffle_for_pzX = shuffle_for_pzX
        self.return_debug = return_debug

        # assert self.conditioning not in ['uniform_prior_pos_delta_l2norm']

        # assert self.conditioning not in ["latent_z_linear", "latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear_internalcond"], "Latent z conditioning does not need a p(z|X) because it's regularized to N(0,1)"

        # Note: 1 DGCNN probably loses some of the rotational invariance between objects
        # assert encoder_type in ["1_dgcnn", "2_dgcnn", "2_vn_dgcnn"]

        self.add_smooth_factor = self.residflow_embnn.add_smooth_factor
        self.division_smooth_factor = self.residflow_embnn.division_smooth_factor
        self.gumbel_temp = self.residflow_embnn.gumbel_temp
        self.encoder_type = encoder_type
        self.sample_z = sample_z
        self.use_action_z = use_action_z
        self.center_feature = self.residflow_embnn.center_feature

        self.dropout_goal_emb = pzX_dropout_goal_emb
        self.latent_z_linear_size = self.residflow_embnn.latent_z_linear_size
        self.latent_z_cond_logvar_limit = (
            self.residflow_embnn.latent_z_cond_logvar_limit
        )
        self.hybrid_cond_logvar_limit = self.residflow_embnn.hybrid_cond_logvar_limit
        self.hybrid_cond_pzX_sample_latent = hybrid_cond_pzX_sample_latent

        if (
            self.conditioning
            in ["hybrid_pos_delta_l2norm", "hybrid_pos_delta_l2norm_internalcond"]
            and self.hybrid_cond_pzX_sample_latent
        ):
            # If sampling we don't need to predict the latents mu and logvar
            self.emb_dims = self.residflow_embnn.emb_dims - self.latent_z_linear_size

        self.embedding_routine = pzX_embedding_routine

        # Embedding networks
        if self.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            if self.encoder_type == "1_dgcnn":
                print(f"--- P(z|X) Using 1 DGCNN ---")
                self.p_z_cond_x_embnn = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.encoder_type == "2_dgcnn":
                print(f"--- P(z|X) Using 2 DGCNN ---")
                self.p_z_cond_x_embnn_action = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
                self.p_z_cond_x_embnn_anchor = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.encoder_type == "2_vn_dgcnn":
                assert self.input_dims == 3, "Only support 3D input for VN_DGCNN"
                print(f"--- P(z|X) Using 2 VN_DGCNN ---")
                args = VNArgs()
                self.p_z_cond_x_embnn_action = VN_DGCNN(
                    args, num_part=self.emb_dims, gc=False
                )
                self.p_z_cond_x_embnn_anchor = VN_DGCNN(
                    args, num_part=self.emb_dims, gc=False
                )
            elif self.encoder_type == "action_vndgcnn_anchor_dgcnn":
                assert self.input_dims == 3, "Only support 3D input for VN_DGCNN"
                print(f"--- P(z|X) Using Action VN_DGCNN and Anchor DGCNN ---")
                args = VNArgs()
                self.p_z_cond_x_embnn_action = VN_DGCNN(
                    args, num_part=self.emb_dims, gc=False
                )
                self.p_z_cond_x_embnn_anchor = DGCNN(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.encoder_type == "2_pn++_msg":
                print(f"--- P(z|X) Using 2 PN++ MSG ---")
                self.p_z_cond_x_embnn_action = PointNet2MSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
                self.p_z_cond_x_embnn_anchor = PointNet2MSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.encoder_type == "2_pn++_ssg":
                print(f"--- P(z|X) Using 2 PN++ SSG ---")
                self.p_z_cond_x_embnn_action = PointNet2SSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
                self.p_z_cond_x_embnn_anchor = PointNet2SSG(
                    num_classes=self.emb_dims, additional_channel=self.input_dims - 3
                )
            elif self.encoder_type == "2_pn++":
                print(f"--- P(z|X) Using 2 PyG PN++ ---")
                args = PN2DenseParams()
                self.p_z_cond_x_embnn_action = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )
                self.p_z_cond_x_embnn_anchor = PN2DenseWrapper(
                    in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args
                )
            else:
                raise ValueError()
        else:
            if self.encoder_type == "1_dgcnn":
                print(f"--- P(z|X) Using 1 DGCNN Classification ---")
                self.p_z_cond_x_embnn = DGCNNClassification(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                )
            elif self.encoder_type == "2_dgcnn":
                print(f"--- P(z|X) Using 2 DGCNN Classification ---")
                self.p_z_cond_x_embnn_action = DGCNNClassification(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    dropout=0.5,
                    output_channels=self.latent_z_linear_size,
                )
                self.p_z_cond_x_embnn_anchor = DGCNNClassification(
                    input_dims=self.input_dims,
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    dropout=0.5,
                    output_channels=self.latent_z_linear_size,
                )
            # TODO: Update architectures for these conditioning types
            else:
                raise ValueError()

        # Set up the transformer
        self.pzX_transformer = pzX_transformer
        self.pzX_transformer_embnn_dims = pzX_transformer_embnn_dims
        self.pzX_transformer_emb_dims = pzX_transformer_emb_dims
        if self.pzX_transformer != "none":
            if self.pzX_transformer == "cross_object":
                print(f"--- P(z|X) Using Cross Object Transformer ---")
                if self.conditioning not in [
                    "latent_z_linear",
                    "latent_z_linear_internalcond",
                ]:
                    if self.encoder_type == "2_dgcnn":
                        print(f"------ With 2 DGCNN Encoders ------")
                        self.p_z_cond_x_embnn_action = DGCNN(
                            input_dims=self.input_dims,
                            emb_dims=self.pzX_transformer_embnn_dims,
                            num_heads=1,
                            last_relu=False,
                        )
                        self.p_z_cond_x_embnn_anchor = DGCNN(
                            input_dims=self.input_dims,
                            emb_dims=self.pzX_transformer_embnn_dims,
                            num_heads=1,
                            last_relu=False,
                        )
                    elif self.encoder_type == "2_vn_dgcnn":
                        assert (
                            self.input_dims == 3
                        ), "Only support 3D input for VN_DGCNN"
                        print(f"------ With 2 VN_DGCNN Encoders ------")
                        args = VNArgs()
                        self.p_z_cond_x_embnn_action = VN_DGCNN(
                            args, num_part=self.pzX_transformer_embnn_dims, gc=False
                        )
                        self.p_z_cond_x_embnn_anchor = VN_DGCNN(
                            args, num_part=self.pzX_transformer_embnn_dims, gc=False
                        )
                    elif self.encoder_type == "action_vndgcnn_anchor_dgcnn":
                        assert (
                            self.input_dims == 3
                        ), "Only support 3D input for VN_DGCNN"
                        print(
                            f"------ With Action VN_DGCNN and Anchor DGCNN Encoders ------"
                        )
                        args = VNArgs()
                        self.p_z_cond_x_embnn_action = VN_DGCNN(
                            args, num_part=self.pzX_transformer_embnn_dims, gc=False
                        )
                        self.p_z_cond_x_embnn_anchor = DGCNN(
                            input_dims=self.input_dims,
                            emb_dims=self.pzX_transformer_embnn_dims,
                            num_heads=1,
                            last_relu=False,
                        )
                    elif self.encoder_type == "2_pn++_msg":
                        print(f"------ With 2 PN++ MSG Encoders ------")
                        self.p_z_cond_x_embnn_action = PointNet2MSG(
                            num_classes=self.pzX_transformer_embnn_dims,
                            additional_channel=self.input_dims - 3,
                        )
                        self.p_z_cond_x_embnn_anchor = PointNet2MSG(
                            num_classes=self.pzX_transformer_embnn_dims,
                            additional_channel=self.input_dims - 3,
                        )
                    elif self.encoder_type == "2_pn++_ssg":
                        print(f"------ With 2 PN++ SSG Encoders ------")
                        self.p_z_cond_x_embnn_action = PointNet2SSG(
                            num_classes=self.pzX_transformer_embnn_dims,
                            additional_channel=self.input_dims - 3,
                        )
                        self.p_z_cond_x_embnn_anchor = PointNet2SSG(
                            num_classes=self.pzX_transformer_embnn_dims,
                            additional_channel=self.input_dims - 3,
                        )
                    elif self.encoder_type == "2_pn++":
                        print(f"------ With 2 PyG PN++ Encoders ------")
                        args = PN2DenseParams()
                        self.p_z_cond_x_embnn_action = PN2DenseWrapper(
                            in_channels=self.input_dims - 3,
                            out_channels=self.pzX_transformer_embnn_dims,
                            p=args,
                        )
                        self.p_z_cond_x_embnn_anchor = PN2DenseWrapper(
                            in_channels=self.input_dims - 3,
                            out_channels=self.pzX_transformer_embnn_dims,
                            p=args,
                        )
                    else:
                        raise ValueError(
                            f"pzX_transformer {self.pzX_transformer} not implemented for encoder_type {self.encoder_type}"
                        )
                elif self.conditioning in [
                    "latent_z_linear",
                    "latent_z_linear_internalcond",
                ]:
                    if self.encoder_type == "1_dgcnn":
                        print(f"------ With 1 DGCNN Classification ------")
                        self.p_z_cond_x_embnn = DGCNNClassification(
                            input_dims=self.input_dims,
                            emb_dims=self.emb_dims,
                            num_heads=self.num_emb_heads,
                        )
                    elif self.encoder_type == "2_dgcnn":
                        print(f"------ With 2 DGCNN Classification ------")
                        self.p_z_cond_x_embnn_action = DGCNNClassification(
                            input_dims=self.input_dims,
                            emb_dims=self.emb_dims,
                            num_heads=self.num_emb_heads,
                            dropout=0.5,
                            output_channels=self.latent_z_linear_size,
                        )
                        self.p_z_cond_x_embnn_anchor = DGCNNClassification(
                            input_dims=self.input_dims,
                            emb_dims=self.emb_dims,
                            num_heads=self.num_emb_heads,
                            dropout=0.5,
                            output_channels=self.latent_z_linear_size,
                        )
                    else:
                        raise ValueError(
                            f"pzX_transformer {self.pzX_transformer} not implemented for encoder_type {self.encoder_type}"
                        )

                self.p_z_cond_x_action_transformer = Transformer(
                    emb_dims=self.pzX_transformer_emb_dims,
                    return_attn=True,
                    bidirectional=False,
                )
                self.p_z_cond_x_anchor_transformer = Transformer(
                    emb_dims=self.pzX_transformer_emb_dims,
                    return_attn=True,
                    bidirectional=False,
                )

                self.proj_emb_dims = self.emb_dims
                if self.conditioning in [
                    "latent_z_linear",
                    "latent_z_linear_internalcond",
                ]:
                    self.proj_emb_dims = 2 * self.latent_z_linear_size
                    self.latent_proj = nn.Sequential(
                        PointNet([self.pzX_transformer_emb_dims, 64, 64, 64, 128, 512]),
                        nn.Conv1d(512, self.proj_emb_dims, kernel_size=1, bias=False),
                    )
                else:
                    self.action_proj = nn.Sequential(
                        PointNet([self.pzX_transformer_emb_dims, 64, 64, 64, 128, 512]),
                        nn.Conv1d(512, self.proj_emb_dims, kernel_size=1, bias=False),
                    )
                    self.anchor_proj = nn.Sequential(
                        PointNet([self.pzX_transformer_emb_dims, 64, 64, 64, 128, 512]),
                        nn.Conv1d(512, self.proj_emb_dims, kernel_size=1, bias=False),
                    )
            else:
                raise ValueError(f"Unknown pzX_transformer {self.pzX_transformer}.")

        # Set up Auxillary networks
        self.embedding_option = pzX_embedding_option
        if self.embedding_routine in [
            "action2anchor",
            "anchor2action",
            "action2anchor2global",
            "anchor2action2global",
        ]:
            assert (
                self.pzX_transformer != "none"
            ), "pzX_transformer must be set for action2anchor or anchor2action"

            self.cond_embnn_type = self.encoder_type
            self.cond_emb_dims = self.pzX_transformer_emb_dims
            self.cond_input_dims = self.input_dims

            if self.cond_embnn_type == "2_dgcnn":
                self.cond_embnn = DGCNN(
                    input_dims=self.cond_input_dims,
                    emb_dims=self.cond_emb_dims,
                    num_heads=1,
                    last_relu=False,
                )
                pzX_net = DGCNN(
                    input_dims=self.input_dims + self.emb_dims,
                    emb_dims=self.pzX_transformer_embnn_dims,
                    num_heads=1,
                    last_relu=False,
                )
            elif self.cond_embnn_type == "2_pn++":
                args = PN2DenseParams()
                self.cond_embnn = PN2DenseWrapper(
                    in_channels=self.cond_input_dims - 3,
                    out_channels=self.cond_emb_dims,
                    p=args,
                )
                pzX_net = PN2DenseWrapper(
                    in_channels=self.input_dims + self.emb_dims - 3,
                    out_channels=self.pzX_transformer_embnn_dims,
                    p=args,
                )
            else:
                raise ValueError(
                    f"cond_embnn_type {self.cond_embnn_type} not implemented for action2anchor/anchor2action embedding"
                )

            if self.embedding_routine in ["action2anchor", "action2anchor2global"]:
                self.p_z_cond_x_embnn_action = pzX_net
            else:
                self.p_z_cond_x_embnn_anchor = pzX_net

            if self.embedding_routine in [
                "action2anchor2global",
                "anchor2action2global",
            ]:
                self.global_mu = nn.Sequential(
                    PointNet(
                        [
                            self.pzX_transformer_emb_dims,
                            self.pzX_transformer_emb_dims // 3,
                            self.pzX_transformer_emb_dims // 6,
                        ]
                    ),
                    nn.Conv1d(
                        self.pzX_transformer_emb_dims // 6,
                        self.latent_z_linear_size,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                self.global_logvar = nn.Sequential(
                    PointNet(
                        [
                            self.pzX_transformer_emb_dims,
                            self.pzX_transformer_emb_dims // 3,
                            self.pzX_transformer_emb_dims // 6,
                        ]
                    ),
                    nn.Conv1d(
                        self.pzX_transformer_emb_dims // 6,
                        self.latent_z_linear_size,
                        kernel_size=1,
                        bias=False,
                    ),
                )

                # Just jointly predict the global latent from TF embeddings
                if self.embedding_option == 0:
                    # No additional networks needed
                    pass
                # Use additional DGCNN + Transformer to get the global latent
                elif self.embedding_option == 1:
                    action_points_and_cond_emb_dims = self.input_dims + self.emb_dims
                    if self.cond_embnn_type == "2_dgcnn":
                        self.action_points_and_cond_embnn = DGCNN(
                            input_dims=action_points_and_cond_emb_dims,
                            emb_dims=self.pzX_transformer_emb_dims,
                            num_heads=1,
                            last_relu=False,
                        )
                        self.anchor_points_and_cond_embnn = DGCNN(
                            input_dims=action_points_and_cond_emb_dims,
                            emb_dims=self.pzX_transformer_emb_dims,
                            num_heads=1,
                            last_relu=False,
                        )
                    else:
                        raise ValueError(
                            f"cond_embnn_type {self.cond_embnn_type} not implemented for embedding_option 1"
                        )

                    self.action_global_emb_transformer = Transformer(
                        emb_dims=self.pzX_transformer_emb_dims,
                        return_attn=True,
                        bidirectional=False,
                    )
                    self.anchor_global_emb_transformer = Transformer(
                        emb_dims=self.pzX_transformer_emb_dims,
                        return_attn=True,
                        bidirectional=False,
                    )
                # Use TF embeddings and Cond. values + pointwise MLP to get the global latent
                elif self.embedding_option == 2:
                    mlp_dims = self.pzX_transformer_emb_dims + self.emb_dims
                    self.action_emb_mlp = nn.Sequential(
                        PointNet([mlp_dims, mlp_dims, mlp_dims]),
                        nn.Conv1d(
                            mlp_dims,
                            self.pzX_transformer_emb_dims,
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                    self.anchor_emb_mlp = nn.Sequential(
                        PointNet([mlp_dims, mlp_dims, mlp_dims]),
                        nn.Conv1d(
                            mlp_dims,
                            self.pzX_transformer_emb_dims,
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                else:
                    raise ValueError(
                        f"Unknown embedding_option {self.embedding_option}"
                    )

    def forward(
        self,
        *input,
        sampling_method="gumbel",
        n_samples=1,
        z_samples=None,
        sample_latent=False,
    ):
        action_points = input[0].permute(0, 2, 1)[
            :, : self.input_dims
        ]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, : self.input_dims]

        # Prepare the action/anchor point clouds
        action_points_dmean = action_points
        anchor_points_dmean = anchor_points
        if self.residflow_embnn.center_feature:
            action_points_dmean = action_points[:, :3] - action_points[:, :3].mean(
                dim=2, keepdim=True
            )
            anchor_points_dmean = anchor_points[:, :3] - anchor_points[:, :3].mean(
                dim=2, keepdim=True
            )

            action_points_dmean = torch.cat(
                [action_points_dmean, action_points[:, 3:]], axis=1
            )
            anchor_points_dmean = torch.cat(
                [anchor_points_dmean, anchor_points[:, 3:]], axis=1
            )

        # Potentially shuffle the action and anchor points
        if self.shuffle_for_pzX:
            action_shuffle_idxs = torch.randperm(action_points_dmean.size()[2])
            anchor_shuffle_idxs = torch.randperm(anchor_points_dmean.size()[2])
            action_points_dmean = action_points_dmean[:, :, action_shuffle_idxs]
            anchor_points_dmean = anchor_points_dmean[:, :, anchor_shuffle_idxs]

        def prepare(arr, is_action):
            if self.shuffle_for_pzX:
                shuffle_idxs = action_shuffle_idxs if is_action else anchor_shuffle_idxs
                return arr[:, :, torch.argsort(shuffle_idxs)]
            else:
                return arr

        # Jointly predict the action and anchor goal embeddings
        if self.embedding_routine == "joint":
            # Obtain the goal embedding
            if self.encoder_type == "1_dgcnn":
                # Jointly predict the action and anchor embeddings
                goal_emb_cond_x = self.p_z_cond_x_embnn(
                    torch.cat([action_points_dmean, anchor_points_dmean], dim=-1)
                )
                goal_emb_cond_x_action = prepare(
                    goal_emb_cond_x[:, :, : action_points_dmean.shape[-1]], True
                )
                goal_emb_cond_x_anchor = prepare(
                    goal_emb_cond_x[:, :, action_points_dmean.shape[-1] :], False
                )
            else:
                # Separately predict the action and anchor embeddings
                action_emb = self.p_z_cond_x_embnn_action(action_points_dmean)
                anchor_emb = self.p_z_cond_x_embnn_anchor(anchor_points_dmean)

                if self.conditioning in [
                    "latent_z_linear",
                    "latent_z_linear_internalcond",
                ]:
                    # Get an action/anchor latent to pass through the transformer
                    action_emb = torch.cat(action_emb, dim=1)
                    anchor_emb = torch.cat(anchor_emb, dim=1)

                # Apply cross-object transformer
                if self.pzX_transformer in ["cross_object"]:
                    action_emb_tf, action_attn = self.p_z_cond_x_action_transformer(
                        action_emb, anchor_emb
                    )
                    anchor_emb_tf, anchor_attn = self.p_z_cond_x_anchor_transformer(
                        anchor_emb, action_emb
                    )

                    if self.conditioning in [
                        "latent_z_linear",
                        "latent_z_linear_internalcond",
                    ]:
                        goal_emb_tf = action_emb_tf + anchor_emb_tf
                        goal_emb_cond_x = self.latent_proj(goal_emb_tf)
                    else:
                        action_emb = self.action_proj(action_emb_tf)
                        anchor_emb = self.anchor_proj(anchor_emb_tf)

                elif self.pzX_transformer in ["none"]:
                    pass
                else:
                    raise ValueError(
                        f"pzX_transformer {self.pzX_transformer} not implemented for encoder_type {self.encoder_type}"
                    )

                # Concatenate the action and anchor embeddings
                if self.num_emb_heads > 1 and self.pzX_transformer not in [
                    "cross_object"
                ]:
                    goal_emb_cond_x = [
                        torch.cat(
                            [prepare(action_head, True), prepare(anchor_head, False)],
                            dim=-1,
                        )
                        for action_head, anchor_head in zip(action_emb, anchor_emb)
                    ]
                else:
                    # If using just a continuous latent, set the p(z|X) goal embedding as a list of mu, logvar
                    if self.conditioning in [
                        "latent_z_linear",
                        "latent_z_linear_internalcond",
                    ]:
                        goal_emb_cond_x = [
                            goal_emb_cond_x[:, : self.latent_z_linear_size],
                            goal_emb_cond_x[:, self.latent_z_linear_size :],
                        ]

                    # If not using just continuous latent, set the p(z|X) goal embedding to the concatenated action and anchor embeddings
                    else:
                        goal_emb_cond_x = torch.cat(
                            [prepare(action_emb, True), prepare(anchor_emb, False)],
                            dim=-1,
                        )

            # Get n samples of spatially conditioned action and anchor points
            if sample_latent:
                embedding_samples = Multimodal_ResidualFlow_DiffEmbTransformer.sample(
                    self,
                    action_points[:, :3],
                    anchor_points[:, :3],
                    goal_emb=goal_emb_cond_x,
                    sampling_method=sampling_method,
                    n_samples=n_samples,
                )
            else:
                embedding_samples = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.add_conditioning(
                        self,
                        goal_emb_cond_x,
                        action_points[:, :3],
                        anchor_points[:, :3],
                        self.conditioning,
                        sampling_method=sampling_method,
                        n_samples=n_samples,
                        z_samples=z_samples,
                    )
                )

        # Get the action and anchor sequentially, action->anchor->goal emb. or anchor->action->goal emb.
        elif self.embedding_routine in [
            "action2anchor",
            "anchor2action",
            "action2anchor2global",
            "anchor2action2global",
        ]:
            # Do action first
            if self.embedding_routine in ["action2anchor", "action2anchor2global"]:
                # Get embeddings
                first_cond_emb = self.cond_embnn(action_points_dmean)
                second_emb = self.p_z_cond_x_embnn_anchor(anchor_points_dmean)

                # Do anchor cross object transformer
                first_emb_tf, first_emb_attn = self.p_z_cond_x_action_transformer(
                    first_cond_emb, second_emb
                )

                # Project to the correct dimension
                first_emb = self.action_proj(first_emb_tf)
            # Do anchor first
            else:
                first_cond_emb = self.cond_embnn(anchor_points_dmean)
                second_emb = self.p_z_cond_x_embnn_action(action_points_dmean)

                # Do action cross object transformer
                first_emb_tf, first_emb_attn = self.p_z_cond_x_anchor_transformer(
                    first_cond_emb, second_emb
                )

                # Project to the correct dimension
                first_emb = self.anchor_proj(first_emb_tf)

            # Get spatially conditioned action
            if sample_latent:
                first_embedding_samples = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.sample_single(
                        self,
                        points=(
                            action_points[:, :3]
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else anchor_points[:, :3]
                        ),
                        goal_emb=first_emb,
                        sample_type=(
                            "action"
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else "anchor"
                        ),
                        sampling_method=sampling_method,
                        n_samples=n_samples,
                    )
                )
            # Get spatially conditioned anchor
            else:
                first_embedding_samples = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.add_single_conditioning(
                        self,
                        goal_emb=first_emb,
                        points=(
                            action_points[:, :3]
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else anchor_points[:, :3]
                        ),
                        conditioning=(
                            self.conditioning
                            if "global" not in self.embedding_routine
                            else "pos_delta_l2norm"
                        ),
                        cond_type=(
                            "action"
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else "anchor"
                        ),
                        sampling_method=sampling_method,
                        n_samples=n_samples,
                        z_samples=z_samples,
                    )
                )

            # Apply spatial conditioning to the first points
            cur_cond_type = (
                "action"
                if self.embedding_routine in ["action2anchor", "action2anchor2global"]
                else "anchor"
            )
            first_points_and_cond = first_embedding_samples[0][
                f"{cur_cond_type}_points_and_cond"
            ]
            first_points_and_cond_centered_xyz = first_points_and_cond[
                :, :3
            ] - first_points_and_cond[:, :3].mean(dim=2, keepdim=True)
            first_points_and_cond_dmean = torch.cat(
                [first_points_and_cond_centered_xyz, first_points_and_cond[:, 3:]],
                axis=1,
            )

            # Pass conditioned first points through pointwise MLP or DGCNN
            if self.embedding_routine in ["action2anchor", "action2anchor2global"]:
                # Get action embeddings
                first_emb2 = self.p_z_cond_x_embnn_action(first_points_and_cond_dmean)
                second_emb_tf, second_attn = self.p_z_cond_x_anchor_transformer(
                    second_emb, first_emb2
                )
                second_emb = self.action_proj(second_emb_tf)

            else:
                # Get anchor embeddings
                first_emb2 = self.p_z_cond_x_embnn_anchor(first_points_and_cond_dmean)
                second_emb_tf, second_attn = self.p_z_cond_x_action_transformer(
                    second_emb, first_emb2
                )
                second_emb = self.anchor_proj(second_emb_tf)

            # Get spatially conditioned second points
            if sample_latent:
                second_embedding_samples = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.sample_single(
                        self,
                        points=(
                            anchor_points[:, :3]
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else action_points[:, :3]
                        ),
                        goal_emb=second_emb,
                        sample_type=(
                            "anchor"
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else "action"
                        ),
                        sampling_method=sampling_method,
                        n_samples=n_samples,
                    )
                )
            else:
                second_embedding_samples = (
                    Multimodal_ResidualFlow_DiffEmbTransformer.add_single_conditioning(
                        self,
                        goal_emb=second_emb,
                        points=(
                            anchor_points[:, :3]
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else action_points[:, :3]
                        ),
                        conditioning=(
                            self.conditioning
                            if "global" not in self.embedding_routine
                            else "pos_delta_l2norm"
                        ),
                        cond_type=(
                            "anchor"
                            if self.embedding_routine
                            in ["action2anchor", "action2anchor2global"]
                            else "action"
                        ),
                        sampling_method=sampling_method,
                        n_samples=n_samples,
                        z_samples=z_samples,
                    )
                )

            if self.embedding_routine in ["action2anchor", "action2anchor2global"]:
                goal_emb_cond_x = torch.cat(
                    [prepare(first_emb, True), prepare(second_emb, False)], dim=-1
                )
                action_embedding_samples = first_embedding_samples
                anchor_embedding_samples = second_embedding_samples
            else:
                goal_emb_cond_x = torch.cat(
                    [prepare(second_emb, True), prepare(first_emb, False)], dim=-1
                )
                action_embedding_samples = second_embedding_samples
                anchor_embedding_samples = first_embedding_samples

            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu

            # Prepare outputs
            embedding_samples = []
            for action_embedding_sample, anchor_embedding_sample in zip(
                action_embedding_samples, anchor_embedding_samples
            ):
                if self.conditioning in ["pos_delta_l2norm"]:
                    embedding_sample = {
                        "action_points_and_cond": action_embedding_sample[
                            "action_points_and_cond"
                        ],
                        "anchor_points_and_cond": anchor_embedding_sample[
                            "anchor_points_and_cond"
                        ],
                        "for_debug": {
                            **action_embedding_sample["for_debug"],
                            **anchor_embedding_sample["for_debug"],
                        },
                    }
                    embedding_samples.append(embedding_sample)
                elif self.conditioning in [
                    "hybrid_pos_delta_l2norm_global",
                    "hybrid_pos_delta_l2norm_global_internalcond",
                ]:
                    embedding_sample = {
                        "action_points_and_cond": action_embedding_sample[
                            "action_points_and_cond"
                        ],
                        "anchor_points_and_cond": anchor_embedding_sample[
                            "anchor_points_and_cond"
                        ],
                        "for_debug": {
                            **action_embedding_sample["for_debug"],
                            **anchor_embedding_sample["for_debug"],
                        },
                    }
                    # If using global latent and not sampling from a Gaussian, calculate the global latent
                    if not sample_latent and "global" in self.embedding_routine:
                        # Just jointly predict the global latent from TF embeddings
                        if self.embedding_option == 0:
                            combined_emb = torch.cat(
                                [first_emb_tf, second_emb_tf], dim=-1
                            )

                            # Sum of average pool and max pool
                            global_emb = (
                                torch.mean(combined_emb, dim=2)
                                + torch.max(combined_emb, dim=2)[0]
                            )

                            global_mu = self.global_mu(global_emb.unsqueeze(2))
                            global_logvar = self.global_logvar(global_emb.unsqueeze(2))
                        # Use additional DGCNN + Transformer to get the global latent
                        elif self.embedding_option == 1:
                            action_points_and_cond = action_embedding_sample[
                                "action_points_and_cond"
                            ]
                            anchor_points_and_cond = anchor_embedding_sample[
                                "anchor_points_and_cond"
                            ]

                            action_points_and_cond_emb = (
                                self.action_points_and_cond_embnn(
                                    action_points_and_cond
                                )
                            )
                            anchor_points_and_cond_emb = (
                                self.anchor_points_and_cond_embnn(
                                    anchor_points_and_cond
                                )
                            )

                            action_points_and_cond_emb_tf, action_attn = (
                                self.action_global_emb_transformer(
                                    action_points_and_cond_emb,
                                    anchor_points_and_cond_emb,
                                )
                            )
                            anchor_points_and_cond_emb_tf, anchor_attn = (
                                self.anchor_global_emb_transformer(
                                    anchor_points_and_cond_emb,
                                    action_points_and_cond_emb,
                                )
                            )

                            combined_emb = torch.cat(
                                [
                                    action_points_and_cond_emb_tf,
                                    anchor_points_and_cond_emb_tf,
                                ],
                                dim=-1,
                            )

                            # Sum of average pool and max pool
                            global_emb = (
                                torch.mean(combined_emb, dim=2)
                                + torch.max(combined_emb, dim=2)[0]
                            )

                            global_mu = self.global_mu(global_emb.unsqueeze(2))
                            global_logvar = self.global_logvar(global_emb.unsqueeze(2))

                        # Use TF embeddings and Cond. values + pointwise MLP to get the global latent
                        elif self.embedding_option == 2:
                            # Concat Cond. values to the TF embeddings
                            action_cond_vals = action_embedding_sample[
                                "action_points_and_cond"
                            ][:, 3:]
                            anchor_cond_vals = anchor_embedding_sample[
                                "anchor_points_and_cond"
                            ][:, 3:]

                            action_emb_tf = (
                                first_emb_tf
                                if self.embedding_routine in ["action2anchor2global"]
                                else second_emb_tf
                            )
                            anchor_emb_tf = (
                                second_emb_tf
                                if self.embedding_routine in ["action2anchor2global"]
                                else first_emb_tf
                            )

                            action_emb_tf_cond = torch.cat(
                                [action_emb_tf, action_cond_vals], dim=1
                            )
                            anchor_emb_tf_cond = torch.cat(
                                [anchor_emb_tf, anchor_cond_vals], dim=1
                            )

                            action_emb_cond = self.action_emb_mlp(action_emb_tf_cond)
                            anchor_emb_cond = self.anchor_emb_mlp(anchor_emb_tf_cond)

                            combined_emb = torch.cat(
                                [action_emb_cond, anchor_emb_cond], dim=-1
                            )

                            global_emb = (
                                torch.mean(combined_emb, dim=2)
                                + torch.max(combined_emb, dim=2)[0]
                            )

                            global_mu = self.global_mu(global_emb.unsqueeze(2))
                            global_logvar = self.global_logvar(global_emb.unsqueeze(2))
                        else:
                            raise ValueError(
                                f"Unknown embedding_option {self.embedding_option}"
                            )

                        global_latent = reparametrize(global_mu, global_logvar)

                        embedding_sample["for_debug"]["goal_emb_mu"] = global_mu
                        embedding_sample["for_debug"]["goal_emb_logvar"] = global_logvar
                        embedding_sample["for_debug"]["goal_emb_latent"] = global_latent

                    # Otherwise use the sampled latents, or independent action/anchor latents (non global)
                    else:
                        embedding_sample["for_debug"]["goal_emb_mu"] = torch.cat(
                            [
                                action_embedding_sample["for_debug"]["action_mu"],
                                anchor_embedding_sample["for_debug"]["anchor_mu"],
                            ],
                            axis=1,
                        )
                        embedding_sample["for_debug"]["goal_emb_logvar"] = torch.cat(
                            [
                                action_embedding_sample["for_debug"]["action_logvar"],
                                anchor_embedding_sample["for_debug"]["anchor_logvar"],
                            ],
                            axis=1,
                        )
                        embedding_sample["for_debug"]["goal_emb_latent"] = torch.cat(
                            [
                                action_embedding_sample["for_debug"]["action_latent"],
                                anchor_embedding_sample["for_debug"]["anchor_latent"],
                            ],
                            axis=1,
                        )

                    embedding_samples.append(embedding_sample)

        else:
            raise ValueError(f"Unknown embedding_routine {self.embedding_routine}")

        # Do the TAXPose forward pass

        outputs = []
        for embedding_sample in embedding_samples:
            action_points_and_cond = embedding_sample["action_points_and_cond"]
            anchor_points_and_cond = embedding_sample["anchor_points_and_cond"]
            for_debug = embedding_sample["for_debug"]

            # Optionally prepare the internal TAXPose DGCNN conditioning
            tax_pose_conditioning_action = None
            tax_pose_conditioning_anchor = None
            if self.conditioning == "latent_z_linear_internalcond":
                tax_pose_conditioning_action = torch.tile(
                    for_debug["goal_emb_sample"], (1, 1, action_points.shape[-1])
                )
                tax_pose_conditioning_anchor = torch.tile(
                    for_debug["goal_emb_sample"], (1, 1, anchor_points.shape[-1])
                )
            if self.conditioning in [
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                tax_pose_conditioning_action = torch.tile(
                    for_debug["goal_emb_latent"], (1, 1, action_points.shape[-1])
                )
                tax_pose_conditioning_anchor = torch.tile(
                    for_debug["goal_emb_latent"], (1, 1, anchor_points.shape[-1])
                )

            # Prepare TAXPose inputs
            if self.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_global",
            ]:
                action_points_and_cond = torch.cat(
                    [
                        action_points_and_cond,
                        torch.tile(
                            for_debug["goal_emb_latent"],
                            (1, 1, action_points.shape[-1]),
                        ),
                    ],
                    axis=1,
                )
                anchor_points_and_cond = torch.cat(
                    [
                        anchor_points_and_cond,
                        torch.tile(
                            for_debug["goal_emb_latent"],
                            (1, 1, anchor_points.shape[-1]),
                        ),
                    ],
                    axis=1,
                )

            # Prepare TAXPose inputs
            if self.taxpose_centering == "mean":
                # Mean center the action and anchor points
                action_center = action_points[:, :3].mean(dim=2, keepdim=True)
                anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
            elif self.taxpose_centering == "z":
                # Center about the selected discrete spatial grounding action/anchor point
                action_center = for_debug["trans_pt_action"][:, :, None]
                anchor_center = for_debug["trans_pt_anchor"][:, :, None]
            else:
                raise ValueError(
                    f"Unknown self.taxpose_centering: {self.taxpose_centering}"
                )

            # Decode spatially conditioned action and anchor points into flows to obtain the goal configuration
            flow_action = self.residflow_embnn.tax_pose(
                action_points_and_cond.permute(
                    0, 2, 1
                ),  # Unpermute to match taxpose forward pass input
                anchor_points_and_cond.permute(
                    0, 2, 1
                ),  # Unpermute to match taxpose forward pass input
                conditioning_action=tax_pose_conditioning_action,
                conditioning_anchor=tax_pose_conditioning_anchor,
                action_center=action_center,
                anchor_center=anchor_center,
            )

            # If the demo is available, get p(z|Y) goal embedding
            pzY_logging = {"goal_emb": None}
            if input[2] is not None:
                # Inputs 2 and 3 are the objects in demo positions
                # If we have access to these, we can run the pzY network
                pzY_results = self.residflow_embnn(
                    *input, sampling_method=sampling_method, n_samples=1
                )
                pzY_logging["goal_emb"] = pzY_results[0]["goal_emb"]

                # Pass through additional logging
                if self.conditioning in [
                    "hybrid_pos_delta_l2norm",
                    "hybrid_pos_delta_l2norm_internalcond",
                ]:
                    pzY_logging["pzY_goal_emb_mu_action"] = pzY_results[0][
                        "goal_emb_mu_action"
                    ]
                    pzY_logging["pzY_goal_emb_mu_anchor"] = pzY_results[0][
                        "goal_emb_mu_anchor"
                    ]
                    pzY_logging["pzY_goal_emb_logvar_action"] = pzY_results[0][
                        "goal_emb_logvar_action"
                    ]
                    pzY_logging["pzY_goal_emb_logvar_anchor"] = pzY_results[0][
                        "goal_emb_logvar_anchor"
                    ]
                    pzY_logging["pzY_action_mu"] = pzY_results[0]["action_mu"]
                    pzY_logging["pzY_anchor_mu"] = pzY_results[0]["anchor_mu"]
                    pzY_logging["pzY_action_logvar"] = pzY_results[0]["action_logvar"]
                    pzY_logging["pzY_anchor_logvar"] = pzY_results[0]["anchor_logvar"]
                # Pass through additional logging
                elif self.conditioning in [
                    "hybrid_pos_delta_l2norm_global",
                    "hybrid_pos_delta_l2norm_global_internalcond",
                ]:
                    pzY_logging["pzY_goal_emb_mu"] = pzY_results[0]["goal_emb_mu"]
                    pzY_logging["pzY_goal_emb_logvar"] = pzY_results[0][
                        "goal_emb_logvar"
                    ]
                elif self.conditioning in [
                    "latent_z_linear",
                    "latent_z_linear_internalcond",
                ]:
                    pzY_logging["pzY_goal_emb_mu"] = pzY_results[0]["goal_emb_mu"]
                    pzY_logging["pzY_goal_emb_logvar"] = pzY_results[0][
                        "goal_emb_logvar"
                    ]

            flow_action = {
                **flow_action,
                "goal_emb_cond_x": goal_emb_cond_x,
                **pzY_logging,
                **for_debug,
            }

            outputs.append(flow_action)

        return outputs
