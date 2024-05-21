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

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
from equivariant_pose_graph.models.multilaterate import MultilaterationHead
from equivariant_pose_graph.models.pointnet2 import PointNet2MSG, PointNet2SSG
from equivariant_pose_graph.models.pointnet2pyg import PN2DenseParams, PN2DenseWrapper
from equivariant_pose_graph.models.relative_encoding import (
    MultiheadRelativeAttentionWrapper,
    RotaryPositionEncoding3D,
)
from equivariant_pose_graph.models.vn_dgcnn import VN_DGCNN, VNArgs
from equivariant_pose_graph.utils.conditioning_utils import gumbel_softmax_topk
from equivariant_pose_graph.utils.sample_utils import (
    sample_uniform_cube,
    sample_uniform_random,
)
from equivariant_pose_graph.utils.visualizations import plot_multi_np


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn='dgcnn'):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - \
            points.mean(dim=2, keepdim=True)
    
        points_embedding = self.emb_nn(
            points_dmean)  # B, emb_dims, num_points

        return points_embedding

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    # src, dst (num_dims, num_points)
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
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

    def forward(self, src, tgt, src_mask, tgt_mask, src_pe=None, tgt_pe=None):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, src_pe, tgt_pe), src_mask,
                           tgt, tgt_mask, src_pe, tgt_pe)

    def encode(self, src, src_mask, src_pe, tgt_pe):
        return self.encoder(self.src_embed(src), src_mask, src_pe, tgt_pe)

    def decode(self, memory, src_mask, tgt, tgt_mask, src_pe, tgt_pe):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src_pe, tgt_pe))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, src_pe, tgt_pe):
        for layer in self.layers:
            x = layer(x, mask, src_pe, tgt_pe)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, src_pe, tgt_pe):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, src_pe, tgt_pe)
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
    def __init__(self, size, dropout=None, relative=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.relative = relative

    def forward(self, x, sublayer):
        sub_out = sublayer(self.norm(x))
        if self.relative:
            sub_out = sub_out[0]
        out = x + self.dropout(sub_out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, relative=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, relative), 2)
        self.size = size

    def forward(self, x, mask, src_pe, tgt_pe):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        out = self.sublayer[1](x, self.feed_forward)
        return out


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, relative=False):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, relative), 3)

    def forward(self, x, memory, src_mask, tgt_mask, src_pe, tgt_pe):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        out = self.sublayer[2](x, self.feed_forward)
        return out


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
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
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
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class Transformer(nn.Module):
    def __init__(self, emb_dims=512, n_blocks=1, dropout=0.0, ff_dims=1024, 
                 n_heads=4, return_attn=False, bidirectional=True, relative=True):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional
        self.relative = relative
        c = copy.deepcopy
        
        if self.relative:
            self_attn = MultiheadRelativeAttentionWrapper(self.emb_dims, self.n_heads, dropout=self.dropout)
        else:
            self_attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        
        cross_attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        
        self.model = EncoderDecoder(
            Encoder(
                EncoderLayer(
                    self.emb_dims, 
                    c(self_attn), 
                    c(ff), 
                    self.dropout, 
                    self.relative
                ), 
                self.N
            ),
            Decoder(
                DecoderLayer(
                    self.emb_dims, 
                    c(self_attn), 
                    c(cross_attn), 
                    c(ff), 
                    self.dropout,
                    self.relative
                ), 
                self.N
            ),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        
        src_pe, tgt_pe = None, None
        if self.relative:
            src_pe = input[2]
            tgt_pe = input[3]
        src_embedding = self.model(
            tgt, src, None, None, src_pe, tgt_pe).transpose(2, 1).contiguous()
        src_attn = self.model.decoder.layers[-1].src_attn.attn

        if(self.bidirectional):
            tgt_embedding = self.model(
                src, tgt, None, None, src_pe, tgt_pe).transpose(2, 1).contiguous()
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn

            if(self.return_attn):
                return src_embedding, tgt_embedding, src_attn, tgt_attn
            return src_embedding, tgt_embedding

        if(self.return_attn):
            return src_embedding, src_attn
        return src_embedding


# Share a model with the other script
from equivariant_pose_graph.models.multimodal_transformer_flow import DGCNN

# class DGCNN(nn.Module):
#     def __init__(self, emb_dims=512, input_dims=3):
#         super(DGCNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_dims*2, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
#         self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(emb_dims)

#     def forward(self, x):
#         batch_size, num_dims, num_points = x.size()
#         x = get_graph_feature(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x1 = x.max(dim=-1, keepdim=True)[0]

#         x = F.relu(self.bn2(self.conv2(x)))
#         x2 = x.max(dim=-1, keepdim=True)[0]

#         x = F.relu(self.bn3(self.conv3(x)))
#         x3 = x.max(dim=-1, keepdim=True)[0]

#         x = F.relu(self.bn4(self.conv4(x)))
#         x4 = x.max(dim=-1, keepdim=True)[0]

#         x = torch.cat((x1, x2, x3, x4), dim=1)

#         x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
#         return x


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
        scores = torch.matmul(action_embedding.transpose(
            2, 1).contiguous(), anchor_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        action_corr = torch.matmul(
            anchor_points, scores.transpose(2, 1).contiguous())

        action_centered = action_points - \
            action_points.mean(dim=2, keepdim=True)

        action_corr_centered = action_corr - \
            action_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(
            action_centered, action_corr_centered.transpose(2, 1).contiguous())

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

        t = torch.matmul(-R, action_points.mean(dim=2, keepdim=True)
                         ) + action_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


class PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(nn.Conv1d(
                layer_dims[j], layer_dims[j+1],
                kernel_size=1, bias=False))
            norms.append(nn.BatchNorm1d(layer_dims[j+1]))

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
            PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(emb_dims//8, 3, kernel_size=1, bias=False),
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
            PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(emb_dims//8, 4, kernel_size=1, bias=False),
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

    def __init__(self, emb_dims=512, output_dims=3, pred_weight=True, residual_on=True, 
                 conditioning_type='old', conditioning_size=0, use_weighted_sum='all', use_selected_point_feature=False):
        super(ResidualMLPHead, self).__init__()
        self.flow_emb_dims = emb_dims
        self.weight_emb_dims = emb_dims
        self.conditioning_type = conditioning_type
        
        self.use_weighted_sum = use_weighted_sum
        if self.use_weighted_sum != 'all':
            self.weighted_sum_k = int(re.match(r'^top_([0-9]*)$', self.use_weighted_sum).group(1))
        
        self.use_selected_point_feature = use_selected_point_feature
        self.emb_mul = 1
        # Adding the feature is not supported when doing weighted sum for all points
        if self.use_selected_point_feature and self.use_weighted_sum != 'all':
            self.emb_mul = 2
        
        # Add conditioning before flow weight predictions
        if self.conditioning_type in ['old-flow_weight', 'flow_fix-flow_weight']:
            self.weight_emb_dims = emb_dims + conditioning_size
        # Predict flow and weight in one head
        elif self.conditioning_type in ['flow_fix-one_flow_head', 'flow_fix-post_encoder_one_flow_head']:
            assert output_dims == 4, "Must predict 4 dimensions for flow and weight"

        if self.flow_emb_dims < 10:
            self.proj_flow = nn.Sequential(
                PointNet([self.flow_emb_dims*self.emb_mul, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )
        else:
            self.proj_flow = nn.Sequential(
                PointNet([self.flow_emb_dims*self.emb_mul, self.flow_emb_dims*self.emb_mul//2, self.flow_emb_dims*self.emb_mul//4, self.flow_emb_dims*self.emb_mul//8]),
                nn.Conv1d(self.flow_emb_dims*self.emb_mul//8, output_dims, kernel_size=1, bias=False),
            )
        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([self.weight_emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([self.weight_emb_dims, self.weight_emb_dims//2, self.weight_emb_dims//4, self.weight_emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

        self.residual_on = residual_on

    def forward(self, *input, scores=None, return_flow_component=False, return_embedding=False, conditioning=None):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]

        if(scores is None):
            if(len(input) <= 4):
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(action_query.transpose(
                2, 1).contiguous(), anchor_key) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        if self.use_weighted_sum == 'all':
            corr_points = torch.matmul(
                anchor_points, scores.transpose(2, 1).contiguous())
        else:
            corr_points_khot, corr_points_onehots = gumbel_softmax_topk(scores.transpose(2, 1), k=self.weighted_sum_k, tau=1, hard=True, dim=1)
            
            selected_khot_scores = scores.transpose(2, 1) * corr_points_khot
            selected_normalized_scores = selected_khot_scores / selected_khot_scores.sum(dim=1, keepdim=True)
            
            corr_points = torch.matmul(anchor_points, selected_normalized_scores.contiguous())

        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        if self.use_selected_point_feature and self.use_weighted_sum != 'all':
            selected_anchor_embeddings = [torch.matmul(anchor_embedding, corr_points_onehot) for corr_points_onehot in corr_points_onehots]
            selected_anchor_embeddings = torch.stack(selected_anchor_embeddings, dim=1)
            mean_selected_anchor_embeddings = torch.mean(selected_anchor_embeddings, dim=1)
            
            embedding = torch.cat([action_embedding, mean_selected_anchor_embeddings], dim=1)
        
        residual_flow = self.proj_flow(embedding)  # B,output_dims,N

        if self.conditioning_type in ['flow_fix-one_flow_head', 'flow_fix-post_encoder_one_flow_head']:
            weight = residual_flow[:, 3:, :]
            residual_flow = residual_flow[:, :3, :]
            
        if self.residual_on:
            flow = residual_flow + corr_flow
        else:
            flow = corr_flow

        if self.pred_weight:
            if self.conditioning_type in ['old-flow_weight', 'flow_fix-flow_weight']:
                weight = self.proj_flow_weight(torch.cat([embedding, conditioning], dim=1))
            elif self.conditioning_type in ['flow_fix-one_flow_head', 'flow_fix-post_encoder_one_flow_head']:
                # Use jointly predicted weight
                pass
            else:
                weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            'full_flow': corr_flow_weight,
            'residual_flow': residual_flow,
            'corr_flow': corr_flow,
            'corr_points': corr_points,
            'scores': scores,
        }

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


class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(self, emb_dims=512, input_dims=3, cycle=True, emb_nn='dgcnn', return_flow_component=False, center_feature=False,
                 inital_sampling_ratio=0.2, pred_weight=True, residual_on=True, freeze_embnn=False, use_transformer_attention=True,
                 conditioning_size=0, multilaterate=False, sample=False, mlat_nkps=100, pred_mlat_weight=False,
                 conditioning_type='old', n_heads=4, flow_head_use_weighted_sum='all', flow_head_use_selected_point_feature=False,
                 post_encoder_input_dims=3, flow_direction="both", ghost_points='none', num_ghost_points=256, ghost_point_radius=0.1,
                 relative_3d_encoding=False):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.cycle = cycle
        self.conditioning_size = conditioning_size
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.use_transformer_attention = use_transformer_attention
        self.n_heads = n_heads
        self.post_encoder_input_dims = post_encoder_input_dims
        self.flow_direction = flow_direction

        self.conditioning_type = conditioning_type # 'old', 'flow_fix', 'old-post_encoder', 'encoder-weight'
        encoder_input_dims = self.input_dims
        head_output_dims = self.input_dims
        transformer_emb_dims = self.emb_dims
        head_emb_dims = self.emb_dims
        head_conditioning_size = 0
        # Use original conditioning with weird conditioning value flow as flow weight
        if self.conditioning_type == 'old':
            pass
        # Use correct conditioning with flow weight head prediction as flow weight
        elif self.conditioning_type == 'flow_fix':
            head_output_dims = 3
        # Only add conditioning after the encoder embedding
        elif self.conditioning_type in ['old-post_encoder', 'flow_fix-post_encoder']:
            encoder_input_dims = self.post_encoder_input_dims
            head_output_dims = 3 if self.conditioning_type == 'flow_fix-post_encoder' else self.input_dims
        # Add conditioning at every pointwise operation, or every pointwise operation after the encoder embedding
        elif self.conditioning_type in ['old-all_pointwise_naive', 'old-post_encoder_all_pointwise', 'flow_fix-all_pointwise_naive', 'flow_fix-post_encoder_all_pointwise']:
            encoder_input_dims = self.post_encoder_input_dims if self.conditioning_type in ['old-post_encoder_all_pointwise', 'flow_fix-post_encoder_all_pointwise'] else self.input_dims
            head_output_dims = 3 if self.conditioning_type in ['flow_fix-all_pointwise_naive', 'flow_fix-post_encoder_all_pointwise'] else self.input_dims
            transformer_emb_dims = self.emb_dims + (self.n_heads * (self.input_dims - 3))
            head_emb_dims = self.emb_dims + ((self.n_heads + 1) * (self.input_dims - 3))
        # Add conditioning at the encoder embedding and for the flow weight head prediction
        elif self.conditioning_type in ['old-flow_weight', 'flow_fix-flow_weight']:
            head_output_dims = 3 if self.conditioning_type == 'flow_fix-flow_weight' else self.input_dims
            head_conditioning_size = self.input_dims - 3
        # Predict flow and weight in one head
        elif self.conditioning_type in ['flow_fix-one_flow_head', 'flow_fix-post_encoder_one_flow_head']:
            encoder_input_dims = self.post_encoder_input_dims if self.conditioning_type == 'flow_fix-post_encoder_one_flow_head' else self.input_dims
            head_output_dims = 4
        else:
            raise ValueError(f"Invalid conditioning type {self.conditioning_type}")

        if emb_nn == 'dgcnn':
            print(f'--- TAXPose Decoder using 2 DGCNN ---')
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims,
                input_dims=encoder_input_dims, 
                conditioning_size=self.conditioning_size
            )
            self.emb_nn_anchor = DGCNN(
                emb_dims=self.emb_dims, 
                input_dims=encoder_input_dims, 
                conditioning_size=self.conditioning_size
            )
        elif emb_nn == 'vn_dgcnn':
            print(f'--- TAXPose Decoder using 2 VN-DGCNN ---')
            assert self.conditioning_type in ['old-post_encoder', 'old-post_encoder_all_pointwise', 'flow_fix-post_encoder', 'flow_fix-post_encoder_all_pointwise'], \
                "VN-DGCNN must use post_encoder conditioning type"
            vn_args = VNArgs()
            self.emb_nn_action = VN_DGCNN(vn_args, num_part=self.emb_dims, gc=False)
            self.emb_nn_anchor = VN_DGCNN(vn_args, num_part=self.emb_dims, gc=False)
        elif emb_nn == 'pn++_ssg':
            print(f'--- TAXPose Decoder using 2 PN++ SSG ---')
            self.emb_nn_action = PointNet2SSG(num_classes=self.emb_dims, additional_channel=encoder_input_dims-3)
            self.emb_nn_anchor = PointNet2SSG(num_classes=self.emb_dims, additional_channel=encoder_input_dims-3)
        elif emb_nn == 'pn++_msg':
            print(f'--- TAXPose Decoder using 2 PN++ MSG ---')
            self.emb_nn_action = PointNet2MSG(num_classes=self.emb_dims, additional_channel=encoder_input_dims-3)
            self.emb_nn_anchor = PointNet2MSG(num_classes=self.emb_dims, additional_channel=encoder_input_dims-3)
        elif emb_nn == 'pn++':
            print(f'--- TAXPose Decoder using 2 PN++ ---')
            args = PN2DenseParams()
            self.emb_nn_action = PN2DenseWrapper(in_channels=encoder_input_dims - 3, out_channels=self.emb_dims, p=args)
            self.emb_nn_anchor = PN2DenseWrapper(in_channels=encoder_input_dims - 3, out_channels=self.emb_dims, p=args)
        elif emb_nn == 'dgcnn_action_pn++_anchor':
            print(f'--- TAXPose Decoder using Action DGCNN and Anchor PN++ ---')
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims,
                input_dims=encoder_input_dims, 
                conditioning_size=self.conditioning_size
            )
            args = PN2DenseParams()
            self.emb_nn_anchor = PN2DenseWrapper(in_channels=encoder_input_dims - 3, out_channels=self.emb_dims, p=args)
        else:
            raise Exception('Not implemented')

        self.relative_3d_encoding = relative_3d_encoding
        if self.relative_3d_encoding:
            print(f'--- TAXPose Decoder using Relative 3D Position Encoding ---')
            self.relative_3d_pe = RotaryPositionEncoding3D(feature_dim=transformer_emb_dims)

        self.transformer_action = Transformer(
            emb_dims=transformer_emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=self.n_heads,
            relative=self.relative_3d_encoding
        )
        self.transformer_anchor = Transformer(
            emb_dims=transformer_emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=self.n_heads,
            relative=self.relative_3d_encoding
        )
        
        self.multilaterate = multilaterate
        self.pred_mlat_weight = pred_mlat_weight
        self.flow_head_use_weighted_sum = flow_head_use_weighted_sum
        self.flow_head_use_selected_point_feature = flow_head_use_selected_point_feature
        if self.multilaterate:
            # TODO: Add support for additional conditioning types
            self.head_action = MultilaterationHead(
                emb_dims=head_emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
                pred_mlat_weight=self.pred_mlat_weight,
            )
            self.head_anchor = MultilaterationHead(
                emb_dims=head_emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
                pred_mlat_weight=self.pred_mlat_weight,
            )
        else:
            self.head_action = ResidualMLPHead(
                emb_dims=head_emb_dims, 
                output_dims=head_output_dims, 
                pred_weight=self.pred_weight, 
                residual_on=self.residual_on, 
                conditioning_type=self.conditioning_type,
                conditioning_size=head_conditioning_size,
                use_weighted_sum=self.flow_head_use_weighted_sum,
                use_selected_point_feature=self.flow_head_use_selected_point_feature
            )
            self.head_anchor = ResidualMLPHead(
                emb_dims=head_emb_dims, 
                output_dims=head_output_dims,
                pred_weight=self.pred_weight, 
                residual_on=self.residual_on,
                conditioning_type=self.conditioning_type,
                conditioning_size=head_conditioning_size,
                use_weighted_sum=self.flow_head_use_weighted_sum,
                use_selected_point_feature=self.flow_head_use_selected_point_feature
            )
            
        if self.conditioning_type.split('-')[-1] in ['post_encoder', 'post_encoder_all_pointwise', 'post_encoder_one_flow_head']:
            emb_dims_cond = self.emb_dims + self.input_dims - self.post_encoder_input_dims
            self.proj_flow_cond_action = nn.Sequential(
                PointNet([emb_dims_cond, emb_dims_cond * 2, emb_dims_cond * 4]),
                nn.Conv1d(emb_dims_cond * 4, self.emb_dims, kernel_size=1, bias=False),
            )
            self.proj_flow_cond_anchor = nn.Sequential(
                PointNet([emb_dims_cond, emb_dims_cond * 2, emb_dims_cond * 4]),
                nn.Conv1d(emb_dims_cond * 4, self.emb_dims, kernel_size=1, bias=False),
            )
            
        self.ghost_points = ghost_points # 'none', 'scene', 'p_center'
        if self.ghost_points != 'none':
            self.num_ghost_points = num_ghost_points
            self.ghost_points_radius = ghost_point_radius
            self.ghost_points_encoder = nn.Sequential(
                PointNet([3, transformer_emb_dims//6, transformer_emb_dims//3, transformer_emb_dims]),
            )
        

    def forward(self, *input, conditioning_action=None, conditioning_anchor=None, action_center=None, anchor_center=None):
        action_points = input[0].permute(0, 2, 1) # B,self.input_dims,num_points
        anchor_points = input[1].permute(0, 2, 1) # B,self.input_dims,num_points

        # TAX-Pose defaults
        if action_center is None:
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
        if anchor_center is None:
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)

        action_points_dmean = torch.cat(
            [
                action_points[:,:3,:] - \
                    action_center,
                action_points[:,3:,:],
            ],
            dim=1
        )
        anchor_points_dmean = torch.cat(
            [
                anchor_points[:,:3,:] - \
                    anchor_center,
                anchor_points[:,3:,:],
            ],
            dim=1
        )
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points 
            
        if self.conditioning_type.split('-')[-1] in ['post_encoder', 'post_encoder_all_pointwise', 'post_encoder_one_flow_head']:
            # If doing post-encoder conditioning, only pass in the points to encoder
            action_points_dmean = action_points_dmean[:, :self.post_encoder_input_dims, :]
            anchor_points_dmean = anchor_points_dmean[:, :self.post_encoder_input_dims, :] 
        
            action_embedding = self.emb_nn_action(action_points_dmean, conditioning=conditioning_action)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, conditioning=conditioning_anchor)
            
            # Add the conditioning to the embedding
            action_embedding_stack = torch.cat([action_embedding, action_points[:, self.post_encoder_input_dims:, :]], dim=1)
            anchor_embedding_stack = torch.cat([anchor_embedding, anchor_points[:, self.post_encoder_input_dims:, :]], dim=1)
            
            action_embedding = self.proj_flow_cond_action(action_embedding_stack)
            anchor_embedding = self.proj_flow_cond_anchor(anchor_embedding_stack)
        else:
            action_embedding = self.emb_nn_action(action_points_dmean, conditioning=conditioning_action)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, conditioning=conditioning_anchor)
        
        if self.freeze_embnn:
            action_embedding = action_embedding.detach()
            anchor_embedding = anchor_embedding.detach()
        
        if self.conditioning_type.split('-')[-1] in ['post_encoder_all_pointwise', 'all_pointwise_naive']:
            # Add conditioning to the point embeddings
            action_embedding = torch.cat([action_embedding, torch.tile(action_points[:, 3:, :], (1, self.n_heads, 1))], dim=1)
            anchor_embedding = torch.cat([anchor_embedding, torch.tile(anchor_points[:, 3:, :], (1, self.n_heads, 1))], dim=1)

        if self.ghost_points == 'p_center':
            ghost_points_centered = sample_uniform_cube(self.ghost_points_radius, int(self.num_ghost_points ** (1/3)))
            ghost_points_centered = (
                torch.tensor(ghost_points_centered, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(action_points_dmean.shape[0], 1, 1)
                .to(action_points_dmean.device)
                .permute(0, 2, 1)
            )

            ghost_points_embedding = self.ghost_points_encoder(ghost_points_centered)
            
            anchor_embedding = torch.cat([anchor_embedding, ghost_points_embedding], dim=-1)
            
            ghost_points = ghost_points_centered + anchor_center

        action_pe, anchor_pe = None, None
        if self.relative_3d_encoding:
            action_pe_xyz = action_points_dmean[:, :3, :].permute(0, 2, 1) # B, N, 3
            anchor_pe_xyz = anchor_points_dmean[:, :3, :].permute(0, 2, 1)
            action_pe = self.relative_3d_pe(action_pe_xyz)
            anchor_pe = self.relative_3d_pe(anchor_pe_xyz)

        # tilde_phi, phi are both B,512,N
        action_embedding_tf, action_attn = \
            self.transformer_action(action_embedding, anchor_embedding, action_pe, anchor_pe)
        anchor_embedding_tf, anchor_attn = \
            self.transformer_anchor(anchor_embedding, action_embedding, anchor_pe, action_pe)

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf
        
        if self.conditioning_type.split('-')[-1] in ['post_encoder_all_pointwise', 'all_pointwise_naive']:
            # Add conditioning to the transformer point embeddings
            action_embedding_tf = torch.cat([action_embedding_tf, action_points[:, 3:, :]], dim=1)
            anchor_embedding_tf = torch.cat([anchor_embedding_tf, anchor_points[:, 3:, :]], dim=1)
        
        if(not self.use_transformer_attention):
            action_attn = None
            anchor_attn = None
        action_attn = action_attn.mean(dim=1)

        head_action_conditioning = None
        head_anchor_conditioning = None
        if self.conditioning_type.split('-')[-1] == 'flow_weight':
            # Save the conditioning for the head network
            head_action_conditioning = action_points[:, 3:, :]
            head_anchor_conditioning = anchor_points[:, 3:, :]
        
        if self.conditioning_type.split('-')[0] == 'flow_fix':
            # Only pass in the XYZ points to the head network
            action_points = action_points[:, :3, :]
            anchor_points = anchor_points[:, :3, :]

        head_action_points = action_points
        head_anchor_points = anchor_points
        if self.ghost_points == 'p_center':
            # plot_multi_np([
            #     head_anchor_points[0].detach().cpu().permute(1, 0).numpy(),
            #     ghost_points[0].detach().cpu().permute(1, 0).numpy(), 
            #     head_action_points[0].detach().cpu().permute(1, 0).numpy()])
            # breakpoint()
            head_anchor_points = torch.cat([head_anchor_points, ghost_points], dim=-1)
        
        outputs = {}
        if self.flow_direction in ["both", "action2anchor"]:
            flow_output_action = self.head_action(
                action_embedding_tf, 
                anchor_embedding_tf,
                head_action_points, 
                head_anchor_points, 
                scores=action_attn, 
                conditioning=head_action_conditioning
            )
            flow_action = flow_output_action['full_flow'].permute(0, 2, 1)
            residual_flow_action = flow_output_action['residual_flow'].permute(0, 2, 1)
            corr_flow_action = flow_output_action['corr_flow'].permute(0, 2, 1)
            
            outputs["flow_action"] = flow_action[:, :action_points.shape[2], :]
            outputs["residual_flow_action"] = residual_flow_action[:, :action_points.shape[2], :]
            outputs["corr_flow_action"] = corr_flow_action[:, :action_points.shape[2], :]
            outputs["corr_points_action"] = flow_output_action['corr_points'][:, :, :action_points.shape[2]]
            outputs["scores_action"] = flow_output_action['scores'][:, :action_points.shape[2], :anchor_points.shape[2]]
            
            if "P_A" in flow_output_action:
                original_points_action = flow_output_action["P_A"].permute(0, 2, 1)
                outputs["original_points_action"] = original_points_action
                outputs["sampled_ixs_action"] = flow_output_action["A_ixs"]

        if self.flow_direction in ["both", "anchor2action"]:
            anchor_attn = anchor_attn.mean(dim=1)
            flow_output_anchor = self.head_anchor(
                anchor_embedding_tf, 
                action_embedding_tf, 
                head_anchor_points, 
                head_action_points, 
                scores=anchor_attn, 
                conditioning=head_anchor_conditioning
            )
            flow_anchor = flow_output_anchor['full_flow'].permute(0, 2, 1)
            residual_flow_anchor = flow_output_anchor['residual_flow'].permute(0, 2, 1)
            corr_flow_anchor = flow_output_anchor['corr_flow'].permute(0, 2, 1)
            
            outputs["flow_anchor"] = flow_anchor[:, :anchor_points.shape[2], :]
            outputs["residual_flow_anchor"] = residual_flow_anchor[:, :anchor_points.shape[2], :]
            outputs["corr_flow_anchor"] = corr_flow_anchor[:, :anchor_points.shape[2], :]
            outputs["corr_points_anchor"] = flow_output_anchor['corr_points'][:, :, :anchor_points.shape[2]]
            outputs["scores_anchor"] = flow_output_anchor['scores'][:, :anchor_points.shape[2], :action_points.shape[2]]
            
            if "P_A" in flow_output_anchor:
                original_points_anchor = flow_output_anchor["P_A"].permute(0, 2, 1)
                outputs["original_points_anchor"] = original_points_anchor
                outputs["sampled_ixs_anchor"] = flow_output_anchor["A_ixs"]
            
        return outputs


class AlignedFrameDecoder(nn.Module):
    def __init__(self, freeze_embnn=False, emb_dims=512, input_dims=3, flow_direction="both", 
                 head_output_type="point", flow_frame="original"):
        super(AlignedFrameDecoder, self).__init__()
        
        self.freeze_embnn = freeze_embnn
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.flow_direction = flow_direction
        self.head_output_type = head_output_type
        self.flow_frame = flow_frame # original or aligned
        
        self.output_dims = 3 # flow/point
        if head_output_type in ["flow", "point"]:
            self.output_dims = 4 # flow/point + weight
        
        print(f'--- {head_output_type.title()} Decoder using 2 DGCNN ---')
        self.emb_nn_action = DGCNN(
            emb_dims=self.emb_dims,
            input_dims=self.input_dims, 
            conditioning_size=0
        )
        self.emb_nn_anchor = DGCNN(
            emb_dims=self.emb_dims,
            input_dims=self.input_dims,  
            conditioning_size=0
        )
        
        self.transformer_action = Transformer(
            emb_dims=self.emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=4
        )
        self.transformer_anchor = Transformer(
            emb_dims=self.emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=4
        )
        
        self.action_flow_head = nn.Sequential(
            PointNet([self.emb_dims, self.emb_dims//2, self.emb_dims//4, self.emb_dims//8]),
            nn.Conv1d(self.emb_dims//8, self.output_dims, kernel_size=1, bias=False),
        )
        self.anchor_flow_head = nn.Sequential(
            PointNet([self.emb_dims, self.emb_dims//2, self.emb_dims//4, self.emb_dims//8]),
            nn.Conv1d(self.emb_dims//8, self.output_dims, kernel_size=1, bias=False),
        )
        
        
    def forward(self, *input, conditioning_action=None, conditioning_anchor=None, action_center=None, anchor_center=None):
        action_points = input[0].permute(0, 2, 1) # B,self.input_dims,num_points
        anchor_points = input[1].permute(0, 2, 1) # B,self.input_dims,num_points

        assert action_center is not None and anchor_center is not None, "Must provide action and anchor center"

        # Center the point clouds about their respective centers
        action_points_dmean = torch.cat(
            [
                action_points[:,:3,:] - \
                    action_center,
                action_points[:,3:,:],
            ],
            dim=1
        )
        anchor_points_dmean = torch.cat(
            [
                anchor_points[:,:3,:] - \
                    anchor_center,
                anchor_points[:,3:,:],
            ],
            dim=1
        )
        
        # Get the embeddings for the action and anchor point clouds
        action_embedding = self.emb_nn_action(action_points_dmean, conditioning=conditioning_action)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, conditioning=conditioning_anchor)
        
        outputs = {}
        # Get the transformer embeddings
        if self.flow_direction in ["both", "action2anchor"]:
            action_embedding_tf, action_attn = self.transformer_action(action_embedding, anchor_embedding)
            action_embedding = action_embedding + action_embedding_tf
            
            if self.head_output_type == "flow":
                action_flow_and_weights = self.action_flow_head(action_embedding)
                
                if self.flow_frame == "aligned":
                    # action_flow_and_weights are already in the aligned frame
                    pass
                elif self.flow_frame == "original":
                    # Transform the flow back to the original frame
                    pred_action_points_aligned_frame = action_points_dmean[:, :3, :] + action_flow_and_weights[:, :3, :]
                    pred_action_points_anchor_frame = pred_action_points_aligned_frame + anchor_center
                    action_flow_anchor_frame = pred_action_points_anchor_frame - action_points[:, :3, :]
                    action_flow_and_weights = torch.cat(
                        [action_flow_anchor_frame, action_flow_and_weights[:, 3:, :]], dim=1
                    )
                
                outputs["flow_action"] = action_flow_and_weights.permute(0, 2, 1)
            elif self.head_output_type == "point":
                action_points_and_weights = self.action_flow_head(action_embedding)
                
                if self.flow_frame == "aligned":
                    induced_flow_aligned_frame = action_points_and_weights[:, :3, :] - action_points_dmean[:, :3, :]
                    action_flow_and_weights = torch.cat(
                        [induced_flow_aligned_frame, action_points_and_weights[:, 3:, :]], dim=1
                    )
                elif self.flow_frame == "original":
                    pred_action_points_anchor_frame = action_points_and_weights[:, :3, :] + anchor_center
                    action_flow_anchor_frame = pred_action_points_anchor_frame - action_points[:, :3, :]
                    action_flow_and_weights = torch.cat(
                        [action_flow_anchor_frame, action_points_and_weights[:, 3:, :]], dim=1
                    )
                    
                outputs["flow_action"] = action_flow_and_weights.permute(0, 2, 1)
            else:
                raise ValueError(f"Invalid head output type {self.head_output_type}")
        
        if self.flow_direction in ["both", "anchor2action"]:
            anchor_embedding_tf, anchor_attn = self.transformer_anchor(anchor_embedding, action_embedding)
            anchor_embedding = anchor_embedding + anchor_embedding_tf
            
            if self.head_output_type == "flow":
                anchor_flow_and_weights = self.anchor_flow_head(anchor_embedding)
                
                if self.flow_frame == "aligned":
                    # anchor_flow_and_weights are already in the aligned frame
                    pass
                elif self.flow_frame == "original":
                    # Transform the flow back to the original frame
                    pred_anchor_points_aligned_frame = anchor_points_dmean[:, :3, :] + anchor_flow_and_weights[:, :3, :]
                    pred_anchor_points_action_frame = pred_anchor_points_aligned_frame + action_center
                    anchor_flow_action_frame = pred_anchor_points_action_frame - anchor_points[:, :3, :]
                    anchor_flow_and_weights = torch.cat(
                        [anchor_flow_action_frame, anchor_flow_and_weights[:, 3:, :]], dim=1
                    )
                
                outputs["flow_anchor"] = anchor_flow_and_weights.permute(0, 2, 1)
            elif self.head_output_type == "point":
                anchor_points_and_weights = self.anchor_flow_head(anchor_embedding)
                
                if self.flow_frame == "aligned":
                    induced_flow_aligned_frame = anchor_points_and_weights[:, :3, :] - anchor_points_dmean[:, :3, :]
                    anchor_flow_and_weights = torch.cat(
                        [induced_flow_aligned_frame, anchor_points_and_weights[:, 3:, :]], dim=1
                    )
                elif self.flow_frame == "original":
                    pred_anchor_points_action_frame = anchor_points_and_weights[:, :3, :] + action_center
                    anchor_flow_action_frame = pred_anchor_points_action_frame - anchor_points[:, :3, :]
                    anchor_flow_and_weights = torch.cat(
                        [anchor_flow_action_frame, anchor_points_and_weights[:, 3:, :]], dim=1
                    )

                outputs["flow_anchor"] = anchor_flow_and_weights.permute(0, 2, 1)
            else:
                raise ValueError(f"Invalid head output type {self.head_output_type}")
        
        if self.flow_direction not in ["both", "action2anchor", "anchor2action"]:
            raise ValueError(f"Invalid flow direction {self.flow_direction}")
        
        return outputs
        