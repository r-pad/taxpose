import copy
import math

import torch
from torch import nn

from taxpose.nets.dgcnn_gc import DGCNN_GC
from taxpose.nets.pointnet import PointNet
from taxpose.utils.se3 import dualflow2pose
from third_party.dcp.model import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
)


class CustomTransformer(nn.Module):
    """This is a custom transformer model that is used to embed the point clouds.
    It is based on the transformer model from the DCP paper.

    See: https://github.com/WangYueFt/dcp/blob/master/model.py
    """

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
        super(CustomTransformer, self).__init__()
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


class ResidualMLPHead(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512, pred_weight=True):
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

        # # Added for debug purpose
        # if return_flow_component:
        #     return {'residual_flow': residual_flow,
        #             'corr_flow': corr_flow,
        #             'full_flow': residual_flow + corr_flow}

        flow = residual_flow + corr_flow
        if self.pred_weight:
            # print("ResidualMLPHead: PRODUCING SVD WEIGHTS!!!!")
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

        return corr_flow_weight


class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=True,
        inital_sampling_ratio=1.0,
        pred_weight=True,
        gc=False,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "pointnet":
            # TODO: Probably want to swap for the original version.
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN_GC(emb_dims=self.emb_dims, gc=gc)
            self.emb_nn_anchor = DGCNN_GC(emb_dims=self.emb_dims, gc=gc)
        # elif emb_nn == "vn":
        #     self.emb_nn_action = VN_PointNet()
        #     self.emb_nn_anchor = VN_PointNet()
        else:
            raise ValueError("Not implemented")
        self.return_flow_component = return_flow_component
        self.center_feature = center_feature
        self.pred_weight = pred_weight

        self.transformer_action = CustomTransformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.transformer_anchor = CustomTransformer(
            emb_dims=emb_dims, return_attn=True, bidirectional=False
        )
        self.head_action = ResidualMLPHead(
            emb_dims=emb_dims, pred_weight=self.pred_weight
        )
        self.head_anchor = ResidualMLPHead(
            emb_dims=emb_dims, pred_weight=self.pred_weight
        )

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)
        if len(input) == 3:
            cat = input[2]
        else:
            cat = None
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        action_embedding = self.emb_nn_action(action_points_dmean, cat)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, cat)

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


def extract_flow_and_weight(x, sigmoid_on=False):
    # x: Batch, num_points, 4
    pred_flow = x[:, :, :3]
    if x.shape[2] > 3:
        if sigmoid_on:
            pred_w = torch.sigmoid(x[:, :, 3])
        else:
            pred_w = x[:, :, 3]
    else:
        pred_w = None
    return pred_flow, pred_w


class BrianChuerAdapter(nn.Module):
    def __init__(self, emb_dims=512, gc=False):
        super().__init__()
        self.model = ResidualFlow_DiffEmbTransformer(emb_dims, gc=gc)
        self.weight_normalize = "l1"
        self.softmax_temperature = None

    def forward(self, X, Y, cat=None):
        Fx, Fy = self.model(X, Y, cat)

        Fx, pred_w_action = extract_flow_and_weight(Fx, True)
        Fy, pred_w_anchor = extract_flow_and_weight(Fy, True)

        pred_T_action = dualflow2pose(
            xyz_src=X,
            xyz_tgt=Y,
            flow_src=Fx,
            flow_tgt=Fy,
            weights_src=pred_w_action,
            weights_tgt=pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
            temperature=self.softmax_temperature,
        )

        # It's weirdly structured...
        mat = pred_T_action.get_matrix().transpose(-1, -2)

        R_pred = mat[:, :3, :3]
        t_pred = mat[:, :3, 3]

        return R_pred, t_pred, pred_T_action, Fx, Fy
