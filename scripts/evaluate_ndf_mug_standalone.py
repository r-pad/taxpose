import os
import os.path as osp
import random
import signal

import hydra
import matplotlib.pyplot as plt
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import numpy as np
import numpy.typing as npt
import omegaconf
import PIL
import pybullet as p
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from airobot import Robot, log_info, log_warn, set_log_level
from airobot.utils import common
from airobot.utils.arm_util import reach_jnt_goal
from airobot.utils.common import euler2quat
from mpl_toolkits.axes_grid1 import ImageGrid
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
from omegaconf import OmegaConf
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Rotate
from rpad.visualize_3d.plots import pointcloud_fig, segmentation_fig
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor

from taxpose.datasets.enums import ObjectClass
from taxpose.datasets.ndf import OBJECT_LABELS_TO_CLASS
from taxpose.datasets.symmetry_utils import (
    gripper_symmetry_labels,
    nonsymmetric_labels,
    rotational_symmetry_labels,
    scalars_to_rgb,
)
from taxpose.nets.pointnet import PointNet

# from taxpose.datasets.ndf import compute_demo_symmetry_features
from taxpose.nets.transformer_flow import CustomTransformer as Transformer
from taxpose.nets.transformer_flow import MultilaterationHead
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer as RF_DET
from taxpose.nets.transformer_flow import ResidualMLPHead, create_embedding_network
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)

# from taxpose.training.equivariance_testing_model2 import EquivarianceTestingModule
from taxpose.utils.compile_result import get_result_df
from taxpose.utils.load_model import get_weights_path
from taxpose.utils.ndf_sim_utils import get_clouds, get_object_clouds
from taxpose.utils.se3 import pure_translation_se3, symmetric_orthogonalization
from third_party.dcp.model import get_graph_feature

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


class DGCNN(nn.Module):
    "Similar to the released one, with input_dims=3"

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
        multilaterate=False,
        mlat_sample: bool = False,
        mlat_nkps: int = 100,
        feature_channels=0,  # Number of extra channels we'll pass into the network.
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.cycle = cycle
        self.feature_channels = feature_channels

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        emb_dims = encoder_cfg.emb_dims

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
        self.head_action: nn.Module
        self.head_anchor: nn.Module
        if multilaterate:
            self.head_action = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=mlat_sample,
                n_kps=mlat_nkps,
            )
            self.head_anchor = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=mlat_sample,
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

        if self.feature_channels > 0:
            # We're basically putting a few MLP layers in on top of the invariant module.
            combined_dims = emb_dims + self.feature_channels
            self.feature_channel_encoder_action = nn.Sequential(
                PointNet([combined_dims, combined_dims * 2, combined_dims * 4]),
                nn.Conv1d(combined_dims * 4, emb_dims, kernel_size=1, bias=False),
            )
            self.feature_channel_encoder_anchor = nn.Sequential(
                PointNet([combined_dims, combined_dims * 2, combined_dims * 4]),
                nn.Conv1d(combined_dims * 4, emb_dims, kernel_size=1, bias=False),
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

        if self.feature_channels > 0:
            # Add a symmetry label to the embeddings.
            action_features = input[2].permute(0, 2, 1)
            anchor_features = input[3].permute(0, 2, 1)

            action_embedding_stack = torch.cat(
                [action_embedding, action_features], axis=1
            )
            anchor_embedding_stack = torch.cat(
                [anchor_embedding, anchor_features], axis=1
            )

            action_embedding = self.feature_channel_encoder_action(
                action_embedding_stack
            )

            anchor_embedding = self.feature_channel_encoder_anchor(
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


def scale_coords(src, scale_len):
    """
    src of shape: B,3,N
    """
    src_zeromined = src - src.min()
    src_normalized = src_zeromined / src_zeromined.max()
    src_scaled = (src_normalized * scale_len).int()
    return src_scaled


mse_criterion = nn.MSELoss(reduction="sum")


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
    # DIFF: WEIRD! There are these pesky 2 extra lines...
    # xyz_src = xyz_src[:, :, :3]
    # xyz_tgt = xyz_tgt[:, :, :3]
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


mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()


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


def compute_inference_symmetry_features(
    points_action: npt.NDArray[np.float32],
    points_anchor: npt.NDArray[np.float32],
    action_class: ObjectClass,
    anchor_class: ObjectClass,
):
    """The only difference between this and compute_demo_symmetry_features is that
    this function RANDOMLY breaks symmetries. This allows the computation
    of the two objects to be independent."""
    assert len(points_action.shape) == 2
    assert len(points_anchor.shape) == 2
    assert points_action.shape[1] == 3
    assert points_anchor.shape[1] == 3

    if anchor_class == ObjectClass.GRIPPER:
        raise ValueError("Anchor class cannot be the gripper.")

    if anchor_class == ObjectClass.MUG or action_class == ObjectClass.MUG:
        # No symmetry.
        action_sym_feats, _ = nonsymmetric_labels(points_action)
        anchor_sym_feats, _ = nonsymmetric_labels(points_anchor)
        anchor_sym_rgb = scalars_to_rgb(anchor_sym_feats[..., 0])
        action_sym_rgb = scalars_to_rgb(action_sym_feats[..., 0])
        return (
            action_sym_feats,
            anchor_sym_feats,
            action_sym_rgb,
            anchor_sym_rgb,
        )

    if action_class == ObjectClass.GRIPPER:
        action_sym_feats, _, _ = gripper_symmetry_labels(points_action)
    elif action_class in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
        # Change here! Notice no input.
        action_sym_feats, _, _, _ = rotational_symmetry_labels(
            points_action, action_class, look_at=None
        )
    else:
        action_sym_feats, _ = nonsymmetric_labels(points_action)

    if anchor_class in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
        anchor_sym_feats, _, _, _ = rotational_symmetry_labels(
            points_anchor, anchor_class, look_at=None
        )
    else:
        anchor_sym_feats, _ = nonsymmetric_labels(points_anchor)

    anchor_sym_rgb = scalars_to_rgb(anchor_sym_feats[..., 0])
    action_sym_rgb = scalars_to_rgb(action_sym_feats[..., 0])

    return (
        action_sym_feats,
        anchor_sym_feats,
        action_sym_rgb,
        anchor_sym_rgb,
    )


def load_data(
    num_points, clouds, classes, action_class, anchor_class, object_type, action
):
    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()
    points_action_mean_np = points_action_np.mean(axis=0)
    points_action_np = points_action_np - points_action_mean_np

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
    points_anchor_np = points_anchor_np - points_action_mean_np
    points_anchor_mean_np = points_anchor_np.mean(axis=0)

    # np.savez(
    #     f"/home/beisner/code/rpad/taxpose/notebooks/data/ndfeval_{action}_data.npz",
    #     points_action_np=points_action_np,
    #     points_anchor_np=points_anchor_np,
    #     action_symmetry_features=action_symmetry_features,
    #     anchor_symmetry_features=anchor_symmetry_features,
    #     action_symmetry_rgb=action_symmetry_rgb,
    # )

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    # rng_state = torch.get_rng_state()
    # torch.manual_seed(123456)
    points_action, points_anchor, ixs_action, ixs_anchor = subsample(
        num_points, points_action, points_anchor
    )

    (
        action_symmetry_features,
        anchor_symmetry_features,
        action_symmetry_rgb,
        anchor_symmetry_rgb,
    ) = compute_inference_symmetry_features(
        points_action[0].numpy(),
        points_anchor[0].numpy(),
        action_class=OBJECT_LABELS_TO_CLASS[(object_type, action_class)],
        anchor_class=OBJECT_LABELS_TO_CLASS[(object_type, anchor_class)],
    )

    # Visualize the symmetry features
    # fig = pointcloud_fig(
    #     points_action_np,
    #     downsample=1,
    #     colors=action_symmetry_rgb,
    # )
    # fig.show()

    # fig = pointcloud_fig(
    #     points_anchor_np,
    #     downsample=1,
    #     colors=anchor_symmetry_rgb,
    # )
    # fig.show()

    action_symmetry_features = (
        torch.from_numpy(action_symmetry_features).float().unsqueeze(0)
    )
    anchor_symmetry_features = (
        torch.from_numpy(anchor_symmetry_features).float().unsqueeze(0)
    )
    action_symmetry_rgb = torch.from_numpy(action_symmetry_rgb).float().unsqueeze(0)
    anchor_symmetry_rgb = torch.from_numpy(anchor_symmetry_rgb).float().unsqueeze(0)

    # assert ixs_action is not None
    # action_symmetry_features = torch.take_along_dim(
    #     action_symmetry_features, ixs_action[..., None], dim=1
    # )
    # anchor_symmetry_features = torch.take_along_dim(
    #     anchor_symmetry_features, ixs_anchor[..., None], dim=1
    # )

    # action_symmetry_rgb = torch.take_along_dim(
    #     action_symmetry_rgb, ixs_action[..., None], dim=1
    # )
    # anchor_symmetry_rgb = torch.take_along_dim(
    #     anchor_symmetry_rgb, ixs_anchor[..., None], dim=1
    # )

    # torch.set_rng_state(rng_state)

    return (
        points_action.cuda(),
        points_anchor.cuda(),
        action_symmetry_features.cuda(),
        anchor_symmetry_features.cuda(),
        action_symmetry_rgb.cuda(),
        anchor_symmetry_rgb.cuda(),
    )


def load_data_raw(num_points, clouds, classes, action_class, anchor_class):
    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    # rng_state = torch.get_rng_state()
    # torch.manual_seed(123456)
    points_action, points_anchor, ixs_action, ixs_anchor = subsample(
        num_points, points_action, points_anchor
    )
    # torch.set_rng_state(rng_state)
    if points_action is None:
        return None, None

    return points_action.cuda(), points_anchor.cuda()


def subsample(num_points, points_action, points_anchor):
    if points_action.shape[1] > num_points:
        points_action, ixs_action = sample_farthest_points(
            points_action, K=num_points, random_start_point=True
        )
    elif points_action.shape[1] < num_points:
        log_info(
            f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})"
        )
        return None, None, None, None
        # raise NotImplementedError(
        #     f'Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})')

    if points_anchor.shape[1] > num_points:
        points_anchor, ixs_anchor = sample_farthest_points(
            points_anchor, K=num_points, random_start_point=True
        )
    elif points_anchor.shape[1] < num_points:
        log_info(
            f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})"
        )
        return None, None, None, None
        # raise NotImplementedError(
        #     f'Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})')

    return points_action, points_anchor, ixs_action, ixs_anchor


def step_for_time(robot, duration, frequency=240):
    for _ in range(int(duration * frequency)):
        robot.pb_client.stepSimulation()


def step_till_goal(robot, goal, max_duration=2.0):
    i = 0

    while not reach_jnt_goal(
        goal,
        robot.arm.get_jpos,
        joint_name=None,
        max_error=robot.arm.cfgs.ARM.MAX_JOINT_ERROR,
    ) and i < int(0.025 * 240):
        i += 1
        robot.pb_client.stepSimulation()


MAX_TIME = 5.0


def create_network(model_cfg):
    network = RF_DET(
        pred_weight=model_cfg.pred_weight,
        encoder_cfg=model_cfg.encoder,
        center_feature=model_cfg.center_feature,
        # inital_sampling_ratio=model_cfg.inital_sampling_ratio,
        residual_on=model_cfg.residual_on,
        multilaterate=model_cfg.multilaterate,
        mlat_sample=model_cfg.mlat_sample,
        mlat_nkps=model_cfg.mlat_nkps,
        feature_channels=model_cfg.feature_channels,
    )
    return network


def create_ndf_networks(hydra_cfg):
    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(), hydra_cfg.model_path + ".pth"
    )
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


def make_grid_fig(eval_dir, grid_img_fn, num_iterations=100):
    img_dir = os.path.join(eval_dir, "teleport_imgs")
    res_file = os.path.join(
        eval_dir, f"trial_{num_iterations-1}/success_rate_eval_implicit.npz"
    )
    results_dict = np.load(res_file)
    fig = plt.figure(figsize=(32.0, 16.0))

    grid = ImageGrid(fig, 111, nrows_ncols=(5, 20), share_all=True, axes_pad=0.3)
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for i, ax in enumerate(range(num_iterations)):
        ax = grid[i]
        ax.imshow(PIL.Image.open(os.path.join(img_dir, f"teleport_{i}.png")))
        ax.set_xlim(350, 550)
        ax.set_ylim(350, 150)

        succ = results_dict["place_success_teleport_list"][i]
        cond = "success" if succ else "fail"
        if "penetration_list" in results_dict:
            max_pen = results_dict["penetration_list"][i]
            title = f"{cond}: {max_pen:0.3f}"
        else:
            title = f"{cond}"
        ax.set_title(title, color="blue" if succ else "red")

    plt.savefig(grid_img_fn)


def load_network_weights(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        artifact = run.use_artifact(checkpoint_reference)
        try:
            checkpoint_path = artifact.get_path("model.ckpt").download(
                root=artifact_dir
            )
        except KeyError:
            # I re-uploaded a few failed runs...
            checkpoint_path = artifact.get_path("last.ckpt").download(root=artifact_dir)

        weights = torch.load(checkpoint_path)["state_dict"]
        # breakpoint()
        # remove "model.emb_nn" prefix from keys
        # weights = {k.replace("model.emb_nn.", ""): v for k, v in weights.items()}
        return weights
    else:
        return torch.load(checkpoint_reference)["state_dict"]


@hydra.main(
    config_path="../configs/",
    config_name="eval_full_mug_standalone",
)
@torch.no_grad()
def main(hydra_cfg):
    print(OmegaConf.to_yaml(hydra_cfg, resolve=True))
    set_log_level("debug" if hydra_cfg.debug else "info")

    run = wandb.init(
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity,
        group=hydra_cfg.wandb.group,
        config=omegaconf.OmegaConf.to_container(
            hydra_cfg, resolve=True, throw_on_missing=True
        ),
        dir=hydra_cfg.wandb.save_dir,
        job_type=hydra_cfg.job_type,
        save_code=True,
    )

    # Configure the output directories.
    eval_save_dir = hydra_cfg.eval_save_dir
    eval_pointclouds_dir = osp.join(eval_save_dir, "pointclouds")
    eval_grasp_imgs_dir = osp.join(eval_save_dir, "grasp_imgs")
    eval_teleport_imgs_dir = osp.join(eval_save_dir, "teleport_imgs")
    util.safe_makedirs(eval_save_dir)
    util.safe_makedirs(eval_pointclouds_dir)
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)

    results_file_name = hydra_cfg.results_file_name
    results_path = osp.join(eval_save_dir, results_file_name)

    obj_class = hydra_cfg.task.name

    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(), obj_class + "_centered_obj_normalized"
    )

    demo_load_dir = osp.join(
        path_util.get_ndf_data(), "demos", obj_class, hydra_cfg.task.demo_exp
    )

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        # vnn_checkpoint_path=vnn_model_path,
    )

    robot = Robot(
        "franka",
        pb_cfg={"gui": hydra_cfg.pybullet_viz, "realtime": False},
        arm_cfg={"self_collision": False, "seed": hydra_cfg.seed},
    )
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(hydra_cfg.seed)
    random.seed(hydra_cfg.seed)
    np.random.seed(hydra_cfg.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    if obj_class != "mug":
        cfg.DEMOS.PLACEMENT_SURFACE = "shelf"
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
    obj_config_name = osp.join(path_util.get_ndf_config(), obj_class + "_obj_cfg.yaml")
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict["shapenet_obj_dir"]
    obj_class = global_dict["object_class"]
    eval_save_dir = global_dict["eval_save_dir"]

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

    # This seems to do nothing... since reset deletes everything.
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

    penetration_list = []

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

    network = create_network(hydra_cfg.model)

    # place_reasoning_module = TAXPoseReasoning(
    #     network,
    #     TAXPoseReasoningConfig(
    #         loop=1,
    #         weight_normalize=hydra_cfg.place_task.weight_normalize,
    #         softmax_temperature=hydra_cfg.place_task.softmax_temperature,
    #     ),
    # )

    # place_model = TAXPoseInferenceModule(place_reasoning_module, symmetry_cfg=None)

    place_model = EquivarianceTrainingModule(
        model=network,
        weight_normalize=hydra_cfg.place_task.weight_normalize,
        softmax_temperature=hydra_cfg.place_task.softmax_temperature,
        sigmoid_on=True,
        flow_supervision="both",
    )

    # place_model = EquivarianceTestingModule(
    #     network,
    #     lr=hydra_cfg.lr,
    #     image_log_period=hydra_cfg.image_logging_period,
    #     weight_normalize=hydra_cfg.place_task.weight_normalize,
    #     softmax_temperature=hydra_cfg.place_task.softmax_temperature,
    #     loop=hydra_cfg.loop,
    #     action="place",
    #     object_type=obj_class,
    #     action_class=hydra_cfg.place_task.action_class,
    #     anchor_class=hydra_cfg.place_task.anchor_class,
    #     normalize_dist=True,
    # )

    place_model.cuda()

    if hydra_cfg.checkpoint_file_place is not None:
        # weights = load_network_weights(
        #     hydra_cfg.checkpoint_file_place, wandb_cfg=hydra_cfg.wandb, run=run
        # )
        # place_model.load_state_dict(weights)
        ckpt_file = get_weights_path(
            hydra_cfg.checkpoint_file_place, hydra_cfg.wandb, run
        )
        try:
            weights = torch.load(ckpt_file)["state_dict"]
            place_model.load_state_dict(weights)
        except RuntimeError:
            # This is an "older" style model, so we need to load the weights
            # manually.
            place_model.model.load_state_dict(weights)

        log_info("Model Loaded from " + str(hydra_cfg.checkpoint_file_place))

    if hydra_cfg.model_eval_on:
        place_model.eval()

    if hydra_cfg.checkpoint_file_place_refinement is not None:
        assert False
        network = create_network(hydra_cfg.model)

        place_model_refinement = EquivarianceTestingModule(
            network,
            lr=hydra_cfg.lr,
            image_log_period=hydra_cfg.image_logging_period,
            weight_normalize=hydra_cfg.place_task.weight_normalize,
            softmax_temperature=hydra_cfg.place_task.softmax_temperature,
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

    network = create_network(hydra_cfg.model)
    # grasp_reasoning_module = TAXPoseReasoning(
    #     network,
    #     TAXPoseReasoningConfig(
    #         loop=1,
    #         weight_normalize=hydra_cfg.grasp_task.weight_normalize,
    #         softmax_temperature=hydra_cfg.grasp_task.softmax_temperature,
    #     ),
    # )

    grasp_model = EquivarianceTrainingModule(
        model=network,
        weight_normalize=hydra_cfg.grasp_task.weight_normalize,
        softmax_temperature=hydra_cfg.grasp_task.softmax_temperature,
        sigmoid_on=True,
        flow_supervision="both",
    )

    # grasp_model = TAXPoseInferenceModule(grasp_reasoning_module, symmetry_cfg=None)
    # grasp_model = EquivarianceTestingModule(
    #     network,
    #     lr=hydra_cfg.lr,
    #     image_log_period=hydra_cfg.image_logging_period,
    #     weight_normalize=hydra_cfg.grasp_task.weight_normalize,
    #     softmax_temperature=hydra_cfg.grasp_task.softmax_temperature,
    #     loop=hydra_cfg.loop,
    #     action="grasp",
    #     object_type=obj_class,
    #     action_class=hydra_cfg.grasp_task.action_class,
    #     anchor_class=hydra_cfg.grasp_task.anchor_class,
    #     normalize_dist=True,
    # )

    grasp_model.cuda()
    if hydra_cfg.model_eval_on:
        grasp_model.eval()

    if hydra_cfg.checkpoint_file_grasp is not None:
        # weights = load_network_weights(
        #     hydra_cfg.checkpoint_file_grasp, wandb_cfg=hydra_cfg.wandb, run=run
        # )
        # grasp_model.load_state_dict(weights)
        ckpt_file = get_weights_path(
            hydra_cfg.checkpoint_file_grasp, hydra_cfg.wandb, run
        )
        try:
            weights = torch.load(ckpt_file)["state_dict"]
            grasp_model.load_state_dict(weights)
        except RuntimeError:
            # This is an "older" style model, so we need to load the weights
            # manually.
            grasp_model.model.load_state_dict(weights)

        log_info("Model Loaded from " + str(hydra_cfg.checkpoint_file_grasp))
    if hydra_cfg.checkpoint_file_grasp_refinement is not None:
        assert False
        network = create_network(hydra_cfg.model)

        grasp_model_refinement = EquivarianceTestingModule(
            network,
            lr=hydra_cfg.lr,
            image_log_period=hydra_cfg.image_logging_period,
            weight_normalize=hydra_cfg.grasp_task.weight_normalize,
            softmax_temperature=hydra_cfg.grasp_task.softmax_temperature,
            loop=hydra_cfg.loop,
            action="grasp",
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
        torch.manual_seed(hydra_cfg.seed + iteration)
        random.seed(hydra_cfg.seed + iteration)
        np.random.seed(hydra_cfg.seed + iteration)

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

        if hydra_cfg.pose_dist.any_pose:
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
        step_for_time(robot, 5.0)

        robot.pb_client.set_step_sim(False)
        robot.arm.move_ee_xyz([0, 0, 0.2])
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
        log_info("any_pose:{}".format(hydra_cfg.pose_dist.any_pose))
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
            # robot.pb_client.set_step_sim(False)

        o_cid = None
        if hydra_cfg.pose_dist.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            # robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)

        # time.sleep(1.5)
        step_for_time(robot, 1.5)

        hide_link(table_id, rack_link_id)

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1])
        )
        viz_dict["start_obj_pose"] = util.pose_stamped2list(obj_pose_world)

        if obj_class == "mug":
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        # time.sleep(1.5)
        step_for_time(robot, 1.5)

        # Open gripper.
        robot.arm.eetool.open()
        step_for_time(robot, 2.0)

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(eval_teleport_imgs_dir, "%d_init.png" % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        points_mug_raw, points_rack_raw = load_data_raw(
            num_points=hydra_cfg.num_points,
            clouds=obj_points,
            classes=obj_classes,
            action_class=0,
            anchor_class=1,
        )
        if points_mug_raw is None:
            continue
        points_gripper_raw, points_mug_raw = load_data_raw(
            num_points=hydra_cfg.num_points,
            clouds=obj_points,
            classes=obj_classes,
            action_class=2,
            anchor_class=0,
        )
        (
            points_mug,
            points_rack,
            features_mug,
            features_rack,
            sym_rgb_mug,
            sym_rgb_rack,
        ) = load_data(
            num_points=hydra_cfg.num_points,
            clouds=obj_points,
            classes=obj_classes,
            action_class=0,
            anchor_class=1,
            object_type=obj_class,
            action="place",
        )

        ans = place_model(
            points_mug, points_rack, features_mug, features_rack
        )  # 1, 4, 4

        if hydra_cfg.pybullet_viz:
            # Create a plotly figure.
            fig = segmentation_fig(
                torch.cat(
                    [
                        points_mug.squeeze(0),
                        points_rack.squeeze(0),
                        ans["pred_points_action"].squeeze(0),
                    ],
                    dim=0,
                )
                .cpu()
                .numpy(),
                torch.cat(
                    [
                        torch.ones(points_mug.shape[1]),
                        2 * torch.ones(points_rack.shape[1]),
                        3 * torch.ones(ans["pred_points_action"].shape[1]),
                    ],
                    dim=0,
                )
                .int()
                .cpu()
                .numpy(),
                labelmap={1: "mug", 2: "rack", 3: "pred"},
            )
            fig.show()

            # Both on the same plot.
            fig = pointcloud_fig(
                torch.cat(
                    [
                        points_mug[0].cpu(),
                        points_rack[0].cpu(),
                        ans["pred_points_action"][0].cpu(),
                    ],
                ),
                downsample=1,
                colors=torch.cat(
                    [
                        sym_rgb_mug[0].cpu(),
                        sym_rgb_rack[0].cpu(),
                        sym_rgb_mug[0].cpu(),
                    ],
                ),
            )
            fig.show()

        if hydra_cfg.checkpoint_file_place_refinement is not None:
            assert False
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
        (
            points_gripper,
            points_mug,
            features_gripper,
            features_mug,
            sym_rgb_gripper,
            sym_rgb_mug,
        ) = load_data(
            num_points=hydra_cfg.num_points,
            clouds=obj_points,
            classes=obj_classes,
            action_class=2,
            anchor_class=0,
            object_type=obj_class,
            action="grasp",
        )
        ans_grasp = grasp_model(
            points_gripper, points_mug, features_gripper, features_mug
        )  # 1, 4, 4
        pred_T_action_init_gripper2mug = ans_grasp["pred_T_action"]
        pred_T_action_mat_gripper2mug = (
            pred_T_action_init_gripper2mug.get_matrix()[0].T.detach().cpu().numpy()
        )
        pred_T_action_mat_gripper2mug[2, -1] -= 0.001

        gripper_relative_pose = get_world_transform(
            pred_T_action_mat_gripper2mug, ee_pose_world, points_gripper_raw
        )  # transform from gripper to mug in world frame
        pre_grasp_ee_pose = util.pose_stamped2list(gripper_relative_pose)

        # Create a plotly figure.
        if hydra_cfg.pybullet_viz:
            fig = segmentation_fig(
                torch.cat(
                    [
                        points_mug.squeeze(0),
                        points_gripper.squeeze(0),
                        ans_grasp["pred_points_action"].squeeze(0),
                    ],
                    dim=0,
                )
                .cpu()
                .numpy(),
                torch.cat(
                    [
                        torch.ones(points_mug.shape[1]),
                        2 * torch.ones(points_gripper.shape[1]),
                        3 * torch.ones(ans_grasp["pred_points_action"].shape[1]),
                    ],
                    dim=0,
                )
                .int()
                .cpu()
                .numpy(),
                labelmap={1: "mug", 2: "gripper", 3: "pred"},
            )
            fig.show()

            # Both on the same plot.
            fig = pointcloud_fig(
                torch.cat(
                    [
                        points_mug[0].cpu(),
                        points_gripper[0].cpu(),
                        ans_grasp["pred_points_action"][0].cpu(),
                    ],
                ),
                downsample=1,
                colors=torch.cat(
                    [
                        sym_rgb_mug[0].cpu(),
                        sym_rgb_gripper[0].cpu(),
                        sym_rgb_gripper[0].cpu(),
                    ],
                ),
            )
            fig.show()

            breakpoint()

        # breakpoint()

        # breakpoint()

        # np.savez(
        #     f"{eval_pointclouds_dir}/{iteration}_init_all_points.npz",
        #     clouds=cloud_points,
        #     colors=cloud_colors,
        #     classes=cloud_classes,
        #     shapenet_id=obj_shapenet_id,
        # )

        np.savez(
            f"{eval_pointclouds_dir}/{iteration}_init_obj_points.npz",
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
        log_info(f"{eval_pointclouds_dir}/{iteration}_init_obj_points.npz")

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
        # robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        # np.savez(
        #     f"{eval_pointclouds_dir}/{iteration}_teleport_all_points.npz",
        #     clouds=cloud_points,
        #     colors=cloud_colors,
        #     classes=cloud_classes,
        #     shapenet_id=obj_shapenet_id,
        # )

        np.savez(
            f"{eval_pointclouds_dir}/{iteration}_teleport_obj_points.npz",
            clouds=obj_points,
            colors=obj_colors,
            classes=obj_classes,
            shapenet_id=obj_shapenet_id,
        )

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, "teleport_%d.png" % iteration
        )
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(
            obj_id, table_id, -1, placement_link_id, enableCollision=True
        )

        # Detect penetration.
        def detect_penetration(thresh=0.001):
            p.performCollisionDetection()
            contacts = p.getContactPoints(obj_id, table_id)
            # breakpoint()
            has_penetration = any([c[8] < -thresh for c in contacts])
            max_penetration = max([-c[8] for c in contacts], default=0.0)
            return has_penetration, max_penetration

        goal_has_penetration, max_penetration = detect_penetration()

        # robot.pb_client.set_step_sim(False)
        # time.sleep(1.0)
        step_for_time(robot, 1.0)

        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        # np.savez(
        #     f"{eval_pointclouds_dir}/{iteration}_post_teleport_all_points.npz",
        #     clouds=cloud_points,
        #     colors=cloud_colors,
        #     classes=cloud_classes,
        #     shapenet_id=obj_shapenet_id,
        # )

        np.savez(
            f"{eval_pointclouds_dir}/{iteration}_post_teleport_obj_points.npz",
            clouds=obj_points,
            colors=obj_colors,
            classes=obj_classes,
            shapenet_id=obj_shapenet_id,
        )

        # time.sleep(1.0)
        step_for_time(robot, 1.0)

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, "post_teleport_%d.png" % iteration
        )
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)

        try:
            obj_surf_contacts = p.getContactPoints(
                obj_id, table_id, -1, placement_link_id
            )
        except:
            breakpoint()
        touching_surf = len(obj_surf_contacts) > 0
        include_penetration = hydra_cfg.include_penetration
        if include_penetration:
            print(
                f"placed around rung: {touching_surf}, goal_has_penetration: {goal_has_penetration}"
            )
            place_success_teleport = touching_surf and not goal_has_penetration
        else:
            place_success_teleport = touching_surf
        place_success_teleport_list.append(place_success_teleport)
        if not place_success_teleport:
            place_fail_teleport_list.append(iteration)

        # time.sleep(1.0)
        step_for_time(robot, 1.0)

        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):
            # reset everything
            # robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
            # if hydra_cfg.pose_dist.any_pose:
            #   robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)

            # time.sleep(0.5)
            step_for_time(robot, 0.5)

            if hydra_cfg.pose_dist.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                # robot.pb_client.set_step_sim(False)

            robot.arm.go_home(ignore_physics=True)
            step_for_time(robot, 5.0)

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
            step_for_time(robot, 2.0)

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
                    # robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    step_till_goal(robot, grasp_jnt_pos, 2.0)

                    # TODO(Ben): Check that the gripper is not in collision...

                    robot.arm.eetool.close(ignore_physics=True)
                    step_for_time(robot, 2.0)

                    # time.sleep(0.2)
                    step_for_time(robot, 0.2)

                    grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(
                        eval_grasp_imgs_dir, "pre_grasp_%d.png" % iteration
                    )
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
                    obj_points, obj_colors, obj_classes = get_object_clouds(cams)

                    # np.savez(
                    #     f"{eval_pointclouds_dir}/{iteration}_pre_grasp_all_points.npz",
                    #     clouds=cloud_points,
                    #     colors=cloud_colors,
                    #     classes=cloud_classes,
                    #     shapenet_id=obj_shapenet_id,
                    # )

                    np.savez(
                        f"{eval_pointclouds_dir}/{iteration}_pre_grasp_obj_points.npz",
                        clouds=obj_points,
                        colors=obj_colors,
                        classes=obj_classes,
                        shapenet_id=obj_shapenet_id,
                    )

                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    cur_jpos = robot.arm.get_jpos()
                    plan1 = ik_helper.plan_joint_motion(
                        cur_jpos, jnt_pos, max_time=MAX_TIME
                    )
                    plan2 = ik_helper.plan_joint_motion(
                        jnt_pos, grasp_jnt_pos, max_time=MAX_TIME
                    )

                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        robot.arm.eetool.open()
                        step_for_time(robot, 0.2)

                        for jnt in plan1:
                            robot.arm.set_jpos(jnt, wait=False)
                            step_for_time(robot, 0.025)

                        robot.arm.set_jpos(plan1[-1], wait=True)
                        step_till_goal(robot, plan1[-1], 1)

                        for jnt in plan2:
                            robot.arm.set_jpos(jnt, wait=False)
                            step_for_time(robot, 0.04)

                        robot.arm.set_jpos(grasp_plan[-1], wait=True)
                        step_till_goal(robot, grasp_plan[-1], 1)

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

                        # time.sleep(0.8)
                        step_for_time(robot, 0.8)

                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[
                            0
                        ]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        step_for_time(robot, 0.2)

                        safeRemoveConstraint(o_cid)
                        # time.sleep(0.8)
                        step_for_time(robot, 0.8)

                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False
                        )
                        # time.sleep(0.8)
                        step_for_time(robot, 0.8)

                        grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                        grasp_img_fname = osp.join(
                            eval_grasp_imgs_dir, "post_grasp_%d.png" % iteration
                        )
                        np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
                        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

                        # np.savez(
                        #     f"{eval_pointclouds_dir}/{iteration}_post_grasp_all_points.npz",
                        #     clouds=cloud_points,
                        #     colors=cloud_colors,
                        #     classes=cloud_classes,
                        #     shapenet_id=obj_shapenet_id,
                        # )

                        np.savez(
                            f"{eval_pointclouds_dir}/{iteration}_post_grasp_obj_points.npz",
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
                                step_for_time(robot, 2)

                                p.resetBasePositionAndOrientation(
                                    obj_id, obj_pos_before_grasp, ori
                                )

                                soft_grasp_close(robot, finger_joint_id, force=40)
                                step_for_time(robot, 2.0)

                                robot.arm.set_jpos(
                                    jnt_pos_before_grasp, ignore_physics=True
                                )
                                step_till_goal(robot, jnt_pos_before_grasp, 5.0)

                                cid = constraint_grasp_close(robot, obj_id)
                                step_for_time(robot, 1)
                                # grasp_rgb = robot.cam.get_images(get_rgb=True)[
                                #     0]
                                # grasp_img_fname = osp.join(
                                #     eval_grasp_imgs_dir, 'after_grasp_success_%d.png' % iteration)
                                # np2img(grasp_rgb.astype(
                                #     np.uint8), grasp_img_fname)
                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = ik_helper.plan_joint_motion(
                                robot.arm.get_jpos(), offset_jnts, max_time=MAX_TIME
                            )

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    robot.arm.set_jpos(jnt, wait=False)
                                    # time.sleep(0.04)
                                    step_for_time(robot, 0.04)

                                robot.arm.set_jpos(offset_plan[-1], wait=True)
                                step_till_goal(robot, offset_plan[-1], 5.0)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False
                        )
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, rack_link_id, enableCollision=False
                        )
                        # time.sleep(1.0)
                        step_for_time(robot, 1.0)

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
                    robot.arm.get_jpos(), pre_place_jnt_pos1, max_time=MAX_TIME
                )
                plan2 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos1, pre_place_jnt_pos2, max_time=MAX_TIME
                )
                plan3 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos2, place_jnt_pos, max_time=MAX_TIME
                )

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        robot.arm.set_jpos(jnt, wait=False)
                        # time.sleep(0.035)
                        step_for_time(robot, 0.035)

                    robot.arm.set_jpos(place_plan[-1], wait=True)
                    step_till_goal(robot, place_plan[-1], 5)

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
                        # time.sleep(0.075)
                        step_for_time(robot, 0.075)

                    robot.arm.set_jpos(plan3[-1], wait=True)
                    step_till_goal(robot, plan3[-1], 5)

                    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    step_for_time(robot, 0.01)

                    robot.arm.eetool.open()
                    step_for_time(robot, 1)

                    # time.sleep(0.2)
                    step_for_time(robot, 0.2)
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        safeCollisionFilterPair(
                            bodyUniqueIdA=robot.arm.robot_id,
                            bodyUniqueIdB=obj_id,
                            linkIndexA=i,
                            linkIndexB=-1,
                            enableCollision=False,
                            physicsClientId=robot.pb_client.get_client_id(),
                        )
                    # This causes nondeterminism. Since it's not really measured for the paper, we can just comment it out.
                    robot.pb_client.set_step_sim(False)
                    robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    robot.pb_client.set_step_sim(True)
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=False
                    )
                    # time.sleep(4.0)
                    step_for_time(robot, 4.0)

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
        step_for_time(robot, 5.0)

        place_success_list.append(place_success)
        grasp_success_list.append(grasp_success)
        if not place_success:
            place_fail_list.append(iteration)
        if not grasp_success:
            grasp_fail_list.append(iteration)
        penetration_list.append(max_penetration)
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
        f = open(results_path, "a")
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
            penetration_list=penetration_list,
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

    # Create a wandb table with the results. We want the following columns:
    final_name = sample_fname
    df = get_result_df(sample_fname, seed=str(hydra_cfg.seed))
    wandb.log({"eval_results": wandb.Table(dataframe=df)})

    make_grid_fig(eval_save_dir, "place_results_fig.png", hydra_cfg.num_iterations)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, util.signal_handler)

    main()
