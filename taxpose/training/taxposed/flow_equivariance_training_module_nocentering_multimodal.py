import re
import time

import numpy as np
import plotly
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import (
    Rotate,
    Transform3d,
    Translate,
    matrix_to_axis_angle,
    random_rotations,
)
from torch import nn
from torchvision.transforms import ToTensor

from taxpose.training.taxposed.point_cloud_training_module import (
    PointCloudTrainingModule,
)
from taxpose.utils.color_utils import color_gradient, get_color
from taxpose.utils.emb_losses import (
    compute_infonce_loss,
    compute_occlusion_infonce_loss,
)
from taxpose.utils.loss_utils import js_div, js_div_mod, wasserstein_distance
from taxpose.utils.se3 import (
    dense_flow_loss,
    dualflow2pose,
    flow2pose,
    get_degree_angle,
    get_translation,
    pure_translation_se3,
)
from taxpose.utils.taxposed_error_metrics import (
    get_2rack_errors,
    get_all_sample_errors,
    get_rpdiff_min_errors,
)

mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()


class EquivarianceTrainingModule(PointCloudTrainingModule):

    def __init__(
        self,
        model=None,
        lr=1e-3,
        image_log_period=500,
        flow_supervision="both",
        point_loss_type=0,
        action_weight=1,
        anchor_weight=1,
        displace_weight=1,
        smoothness_weight=0.1,
        consistency_weight=1,
        latent_weight=0.1,
        vae_reg_loss_weight=0.01,
        rotation_weight=0,
        chamfer_weight=10000,
        return_flow_component=False,
        weight_normalize="l1",
        sigmoid_on=False,
        softmax_temperature=None,
        min_err_across_racks_debug=False,
        error_mode_2rack="batch_min_rack",
        n_samples=1,
        get_errors_across_samples=False,
        use_debug_sampling_methods=False,
        plot_encoder_distribution=False,
        joint_infonce_loss_weight=0.0,
        spatial_distance_regularization_type="demo",
        spatial_distance_regularization_weight=0.0,
        hybrid_cond_regularize_all=False,
        pzY_taxpose_infonce_loss_weight=0,
        pzY_taxpose_occ_infonce_loss_weight=0,
        decoder_type="taxpose",
        flow_frame="original",
        compute_rpdiff_min_errors=True,
        rpdiff_descriptions_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
        )
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period

        self.flow_supervision = flow_supervision
        # 0 for mse loss, 1 for chamfer distance, 2 for mse loss + chamfer distance
        self.point_loss_type = point_loss_type
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.displace_weight = displace_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        self.chamfer_weight = chamfer_weight
        self.rotation_weight = rotation_weight
        self.latent_weight = latent_weight

        # This is only used when the internal model uses self.model.conditioning == "latent_z" or latent_z_1pred or latent_z_1pred_10d
        self.vae_reg_loss_weight = vae_reg_loss_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        self.min_err_across_racks_debug = min_err_across_racks_debug
        self.error_mode_2rack = error_mode_2rack
        self.n_samples = n_samples
        self.get_errors_across_samples = get_errors_across_samples
        self.use_debug_sampling_methods = use_debug_sampling_methods
        self.plot_encoder_distribution = plot_encoder_distribution
        self.joint_infonce_loss_weight = joint_infonce_loss_weight

        self.spatial_distance_regularization_type = spatial_distance_regularization_type
        self.spatial_distance_regularization_weight = (
            spatial_distance_regularization_weight
        )
        if self.weight_normalize == "l1":
            assert self.sigmoid_on, "l1 weight normalization need sigmoid on"

        self.hybrid_cond_regularize_all = hybrid_cond_regularize_all
        self.pzY_taxpose_infonce_loss_weight = pzY_taxpose_infonce_loss_weight
        self.pzY_taxpose_occ_infonce_loss_weight = pzY_taxpose_occ_infonce_loss_weight

        self.get_sample_errors = True
        self.decoder_type = decoder_type

        self.compute_rpdiff_min_errors = compute_rpdiff_min_errors
        self.rpdiff_descriptions_path = rpdiff_descriptions_path

        self.flow_frame = flow_frame  # "original" or "aligned"

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

    def get_transform(
        self,
        points_trans_action,
        points_trans_anchor,
        points_onetrans_action=None,
        points_onetrans_anchor=None,
        mode="forward",
        sampling_method="gumbel",
        n_samples=1,
    ):
        model_outputs = self.model(
            points_trans_action,
            points_trans_anchor,
            points_onetrans_action,
            points_onetrans_anchor,
            mode=mode,
            sampling_method=sampling_method,
            n_samples=n_samples,
        )

        ans_dicts = []
        for i in range(n_samples):
            model_output = model_outputs[i]

            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]

            ans_dict = self.predict(
                model_output=model_output,
                points_trans_action=points_trans_action,
                points_trans_anchor=points_trans_anchor,
            )

            ans_dict["flow_components"] = model_output
            ans_dicts.append(ans_dict)
        return ans_dicts

    def predict(self, model_output, points_trans_action, points_trans_anchor):

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_trans_anchor = points_trans_anchor

        # Get predicted transform
        if self.flow_supervision == "both":
            # Extract the predicted flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action,
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
        elif self.flow_supervision == "action2anchor":
            # Extract the predicted flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
        elif self.flow_supervision == "anchor2action":
            # Extract the predicted flow and weights
            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(
                f"ERROR: Invalid flow supervision type: {self.flow_supervision}"
            )

        if self.decoder_type == "taxpose":
            pred_points_action = pred_T_action.transform_points(points_trans_action)
        elif self.decoder_type in ["flow", "point"] and self.flow_frame == "aligned":
            action_center = model_output["trans_pt_action"][:, :, None]
            anchor_center = model_output["trans_pt_anchor"][:, :, None]

            points_trans_action_in_aligned_frame = (
                points_trans_action - action_center.permute(0, 2, 1)
            )
            points_action_pred_in_aligned_frame = pred_T_action.transform_points(
                points_trans_action_in_aligned_frame
            )
            pred_points_action = (
                points_action_pred_in_aligned_frame + anchor_center.permute(0, 2, 1)
            )
        else:
            raise ValueError(f"ERROR: Invalid decoder type: {self.decoder_type}")

        return {
            "pred_T_action": pred_T_action,
            "pred_points_action": pred_points_action,
        }

    def compute_loss(
        self, model_outputs, batch, log_values={}, loss_prefix="", heads_list=None
    ):
        points_action = batch["points_action"][:, :, :3]
        points_anchor = batch["points_anchor"][:, :, :3]
        points_trans_action = batch["points_action_trans"][:, :, :3]
        points_trans_anchor = batch["points_anchor_trans"][:, :, :3]

        N = points_action.shape[1]

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])
        T_aug_list = (
            [Transform3d(matrix=T_aug) for T_aug in batch["T_aug_list"]]
            if "T_aug_list" in batch
            else None
        )

        R0_max, R0_min, R0_mean = get_degree_angle(T0)
        R1_max, R1_min, R1_mean = get_degree_angle(T1)
        t0_max, t0_min, t0_mean = get_translation(T0)
        t1_max, t1_min, t1_mean = get_translation(T1)

        n_samples = len(model_outputs)

        if heads_list == None:
            heads_list = [None] * n_samples

        total_loss = 0
        for i, (model_output, heads) in enumerate(zip(model_outputs, heads_list)):
            goal_emb = model_output["goal_emb"]

            # If we've applied some sampling, we need to extract the predictions too...
            if "sampled_ixs_action" in model_output:
                ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
                points_action = torch.take_along_dim(points_action, ixs_action, dim=1)
                points_trans_action = torch.take_along_dim(
                    points_trans_action, ixs_action, dim=1
                )

            if "sampled_ixs_anchor" in model_output:
                ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
                points_anchor = torch.take_along_dim(points_anchor, ixs_anchor, dim=1)
                points_trans_anchor = torch.take_along_dim(
                    points_trans_anchor, ixs_anchor, dim=1
                )

            if self.flow_supervision == "both":
                # Extract the flow and weights
                x_action = model_output["flow_action"]
                pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

                x_anchor = model_output["flow_anchor"]
                pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

                # Extract the predicted transform from the bidirectional flow and weights
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

                if self.decoder_type in ["taxpose"]:
                    # Get induced flow and the transformed action points
                    induced_flow_action = (
                        pred_T_action.transform_points(points_trans_action)
                        - points_trans_action
                    ).detach()
                    pred_points_action = pred_T_action.transform_points(
                        points_trans_action
                    )

                    # Get the ground truth transform
                    # pred_T_action=T1T0^-1
                    gt_T_action = T0.inverse().compose(T1)
                    points_action_target = T1.transform_points(points_action)

                    # Action losses
                    # Loss associated with ground truth transform
                    point_loss_action = mse_criterion(
                        pred_points_action,
                        points_action_target,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_action = mse_criterion(
                        pred_flow_action,
                        induced_flow_action,
                    )
                    dense_loss_action = dense_flow_loss(
                        points=points_trans_action,
                        flow_pred=pred_flow_action,
                        trans_gt=gt_T_action,
                    )

                    # Anchor losses
                    pred_T_anchor = pred_T_action.inverse()

                    # Get the induced flow and the transformed anchor points
                    induced_flow_anchor = (
                        pred_T_anchor.transform_points(points_trans_anchor)
                        - points_trans_anchor
                    ).detach()
                    pred_points_anchor = pred_T_anchor.transform_points(
                        points_trans_anchor
                    )

                    # Get the ground truth transform
                    gt_T_anchor = T1.inverse().compose(T0)
                    points_anchor_target = T0.transform_points(points_anchor)

                    # Loss associated with ground truth transform
                    point_loss_anchor = mse_criterion(
                        pred_points_anchor,
                        points_anchor_target,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_anchor = mse_criterion(
                        pred_flow_anchor,
                        induced_flow_anchor,
                    )
                    dense_loss_anchor = dense_flow_loss(
                        points=points_trans_anchor,
                        flow_pred=pred_flow_anchor,
                        trans_gt=gt_T_anchor,
                    )
                elif (
                    self.decoder_type in ["flow", "point"]
                    and self.flow_frame == "aligned"
                ):
                    action_center = model_output["trans_pt_action"][:, :, None]
                    anchor_center = model_output["trans_pt_anchor"][:, :, None]

                    points_trans_action_in_aligned_frame = (
                        torch.cat(  # action points centered about sampled action p
                            [
                                points_trans_action[:, :, :3]
                                - action_center.permute(0, 2, 1),
                                points_trans_action[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )
                    points_trans_anchor_in_aligned_frame = (
                        torch.cat(  # anchor points centered about sampled anchor p'
                            [
                                points_trans_anchor[:, :, :3]
                                - anchor_center.permute(0, 2, 1),
                                points_trans_anchor[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )
                    # Now the points are in a shared coordinate frame that is centered about the sampled points p, p'

                    gt_T_action = T0.inverse().compose(T1)
                    points_action_target = T1.transform_points(
                        points_action
                    )  # Mug in goal position on the transformed anchor

                    points_action_target_in_aligned_frame = torch.cat(  # Goal action points centered about sampled anchor p'
                        [
                            points_action_target[:, :, :3]
                            - anchor_center.permute(0, 2, 1),
                            points_action_target[:, :, 3:],
                        ],
                        dim=-1,
                    )

                    # Apply the predicted transform to the action points in the centered about sampled p, p' frame
                    points_action_pred_in_aligned_frame = (
                        pred_T_action.transform_points(
                            points_trans_action_in_aligned_frame
                        )
                    )

                    # Loss associated with ground truth transform
                    point_loss_action = mse_criterion(
                        points_action_pred_in_aligned_frame,
                        points_action_target_in_aligned_frame,
                    )

                    induced_flow_action = (
                        points_action_pred_in_aligned_frame
                        - points_trans_action_in_aligned_frame
                    ).detach()

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_action = mse_criterion(
                        pred_flow_action,
                        induced_flow_action,
                    )

                    translate_to_action_center = Translate(-action_center.squeeze(-1))
                    translate_to_anchor_center = Translate(-anchor_center.squeeze(-1))

                    gt_T_action_in_aligned_frame = (
                        translate_to_action_center.inverse()
                        .compose(gt_T_action)
                        .compose(translate_to_anchor_center)
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    dense_loss_action = dense_flow_loss(
                        points=points_trans_action_in_aligned_frame,
                        flow_pred=pred_flow_action,
                        trans_gt=gt_T_action_in_aligned_frame,
                    )

                    # Anchor losses
                    pred_T_anchor = pred_T_action.inverse()

                    gt_T_anchor = T1.inverse().compose(T0)
                    points_anchor_target = T0.transform_points(points_anchor)

                    points_anchor_target_in_aligned_frame = (
                        torch.cat(  # Goal anchor points centered about sampled action p
                            [
                                points_anchor_target[:, :, :3]
                                - action_center.permute(0, 2, 1),
                                points_anchor_target[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )

                    # Apply the predicted transform to the anchor points in the centered about sampled p, p' frame
                    points_anchor_pred_in_aligned_frame = (
                        pred_T_anchor.transform_points(
                            points_trans_anchor_in_aligned_frame
                        )
                    )

                    point_loss_anchor = mse_criterion(
                        points_anchor_pred_in_aligned_frame,
                        points_anchor_target_in_aligned_frame,
                    )

                    induced_flow_anchor = (
                        points_anchor_pred_in_aligned_frame
                        - points_anchor_target_in_aligned_frame
                    ).detach()

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_anchor = mse_criterion(
                        pred_flow_anchor,
                        induced_flow_anchor,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    gt_T_anchor_in_aligned_frame = (
                        translate_to_anchor_center.inverse()
                        .compose(gt_T_anchor)
                        .compose(translate_to_action_center)
                    )

                    dense_loss_anchor = dense_flow_loss(
                        points=points_trans_anchor_in_aligned_frame,
                        flow_pred=pred_flow_anchor,
                        trans_gt=gt_T_anchor_in_aligned_frame,
                    )

                else:
                    raise ValueError(
                        f"ERROR: Invalid decoder type: {self.decoder_type}"
                    )

                self.action_weight = (self.action_weight) / (
                    self.action_weight + self.anchor_weight
                )
                self.anchor_weight = (self.anchor_weight) / (
                    self.action_weight + self.anchor_weight
                )
            elif self.flow_supervision == "action2anchor":
                # Extract the flow and weights
                x_action = model_output["flow_action"]
                pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

                # Extract the predicted transform from the action->anchor flow and weights
                pred_T_action = flow2pose(
                    xyz=points_trans_action,
                    flow=pred_flow_action,
                    weights=pred_w_action,
                    return_transform3d=True,
                    normalization_scehme=self.weight_normalize,
                    temperature=self.softmax_temperature,
                )

                if self.decoder_type in ["taxpose"]:
                    # Action losses
                    # Get induced flow and the transformed action points
                    induced_flow_action = (
                        pred_T_action.transform_points(points_trans_action)
                        - points_trans_action
                    ).detach()
                    pred_points_action = pred_T_action.transform_points(
                        points_trans_action
                    )

                    # Get the ground truth transform
                    # pred_T_action=T1T0^-1
                    gt_T_action = T0.inverse().compose(T1)
                    points_action_target = T1.transform_points(points_action)

                    # Loss associated with ground truth transform
                    point_loss_action = mse_criterion(
                        pred_points_action,
                        points_action_target,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_action = mse_criterion(
                        pred_flow_action,
                        induced_flow_action,
                    )
                    dense_loss_action = dense_flow_loss(
                        points=points_trans_action,
                        flow_pred=pred_flow_action,
                        trans_gt=gt_T_action,
                    )
                elif (
                    self.decoder_type in ["flow", "point"]
                    and self.flow_frame == "aligned"
                ):
                    action_center = model_output["trans_pt_action"][:, :, None]
                    anchor_center = model_output["trans_pt_anchor"][:, :, None]

                    points_trans_action_in_aligned_frame = (
                        torch.cat(  # action points centered about sampled action p
                            [
                                points_trans_action[:, :, :3]
                                - action_center.permute(0, 2, 1),
                                points_trans_action[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )
                    # Now the points are in a shared coordinate frame that is centered about the sampled points p, p'

                    gt_T_action = T0.inverse().compose(T1)
                    points_action_target = T1.transform_points(
                        points_action
                    )  # Mug in goal position on the transformed anchor

                    points_action_target_in_aligned_frame = torch.cat(  # Goal action points centered about sampled anchor p'
                        [
                            points_action_target[:, :, :3]
                            - anchor_center.permute(0, 2, 1),
                            points_action_target[:, :, 3:],
                        ],
                        dim=-1,
                    )

                    # Apply the predicted transform to the action points in the centered about sampled p, p' frame
                    points_action_pred_in_aligned_frame = (
                        pred_T_action.transform_points(
                            points_trans_action_in_aligned_frame
                        )
                    )

                    # Loss associated with ground truth transform
                    point_loss_action = mse_criterion(
                        points_action_pred_in_aligned_frame,
                        points_action_target_in_aligned_frame,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    induced_flow_action = (
                        points_action_pred_in_aligned_frame
                        - points_trans_action_in_aligned_frame
                    ).detach()

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_action = mse_criterion(
                        pred_flow_action,
                        induced_flow_action,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    translate_to_action_center = Translate(-action_center.squeeze(-1))
                    translate_to_anchor_center = Translate(-anchor_center.squeeze(-1))

                    gt_T_action_in_aligned_frame = (
                        translate_to_action_center.inverse()
                        .compose(gt_T_action)
                        .compose(translate_to_anchor_center)
                    )

                    dense_loss_action = dense_flow_loss(
                        points=points_trans_action_in_aligned_frame,
                        flow_pred=pred_flow_action,
                        trans_gt=gt_T_action_in_aligned_frame,
                    )
                else:
                    raise ValueError(
                        f"ERROR: Invalid decoder type: {self.decoder_type}"
                    )

                # Anchor losses
                self.anchor_weight = 0
                self.action_weight = 1
                point_loss_anchor = 0
                smoothness_loss_anchor = 0
                dense_loss_anchor = 0
            elif self.flow_supervision == "anchor2action":
                # Extract the flow and weights
                x_anchor = model_output["flow_anchor"]
                pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

                # Extract the predicted transform from the anchor->action flow and weights
                pred_T_anchor = flow2pose(
                    xyz=points_trans_anchor,
                    flow=pred_flow_anchor,
                    weights=pred_w_anchor,
                    return_transform3d=True,
                    normalization_scehme=self.weight_normalize,
                    temperature=self.softmax_temperature,
                )

                if self.decoder_type in ["taxpose"]:
                    # Anchor losses
                    # Get the induced flow and the transformed anchor points
                    induced_flow_anchor = (
                        pred_T_anchor.transform_points(points_trans_anchor)
                        - points_trans_anchor
                    ).detach()
                    pred_points_anchor = pred_T_anchor.transform_points(
                        points_trans_anchor
                    )

                    # Get the ground truth transform
                    gt_T_anchor = T1.inverse().compose(T0)
                    points_anchor_target = T0.transform_points(points_anchor)

                    # Loss associated with ground truth transform
                    point_loss_anchor = mse_criterion(
                        pred_points_anchor,
                        points_anchor_target,
                    )

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_anchor = mse_criterion(
                        pred_flow_anchor,
                        induced_flow_anchor,
                    )
                    dense_loss_anchor = dense_flow_loss(
                        points=points_trans_anchor,
                        flow_pred=pred_flow_anchor,
                        trans_gt=gt_T_anchor,
                    )
                elif (
                    self.decoder_type in ["flow", "point"]
                    and self.flow_frame == "aligned"
                ):
                    action_center = model_output["trans_pt_action"][:, :, None]
                    anchor_center = model_output["trans_pt_anchor"][:, :, None]

                    points_trans_anchor_in_aligned_frame = (
                        torch.cat(  # anchor points centered about sampled anchor p'
                            [
                                points_trans_anchor[:, :, :3]
                                - anchor_center.permute(0, 2, 1),
                                points_trans_anchor[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )
                    # Now the points are in a shared coordinate frame that is centered about the sampled points p, p'

                    # Anchor losses
                    pred_T_anchor = pred_T_action.inverse()

                    gt_T_anchor = T1.inverse().compose(T0)
                    points_anchor_target = T0.transform_points(points_anchor)

                    points_anchor_target_in_aligned_frame = (
                        torch.cat(  # Goal anchor points centered about sampled action p
                            [
                                points_anchor_target[:, :, :3]
                                - action_center.permute(0, 2, 1),
                                points_anchor_target[:, :, 3:],
                            ],
                            dim=-1,
                        )
                    )

                    # Apply the predicted transform to the anchor points in the centered about sampled p, p' frame
                    points_anchor_pred_in_aligned_frame = (
                        pred_T_anchor.transform_points(
                            points_trans_anchor_in_aligned_frame
                        )
                    )

                    point_loss_anchor = mse_criterion(
                        points_anchor_pred_in_aligned_frame,
                        points_anchor_target_in_aligned_frame,
                    )

                    induced_flow_anchor = (
                        points_anchor_pred_in_aligned_frame
                        - points_anchor_target_in_aligned_frame
                    ).detach()

                    # Loss associated flow vectors matching a consistent rigid transform
                    smoothness_loss_anchor = mse_criterion(
                        pred_flow_anchor,
                        induced_flow_anchor,
                    )

                    translate_to_action_center = Translate(-action_center.squeeze(-1))
                    translate_to_anchor_center = Translate(-anchor_center.squeeze(-1))

                    # Loss associated flow vectors matching a consistent rigid transform
                    gt_T_anchor_in_aligned_frame = (
                        translate_to_anchor_center.inverse()
                        .compose(gt_T_anchor)
                        .compose(translate_to_action_center)
                    )

                    dense_loss_anchor = dense_flow_loss(
                        points=points_trans_anchor_in_aligned_frame,
                        flow_pred=pred_flow_anchor,
                        trans_gt=gt_T_anchor_in_aligned_frame,
                    )
                else:
                    raise ValueError(
                        f"ERROR: Invalid decoder type: {self.decoder_type}"
                    )

                # Action losses
                pred_T_action = pred_T_anchor.inverse()
                self.action_weight = 0
                self.anchor_weight = 1
                point_loss_action = 0
                smoothness_loss_action = 0
                dense_loss_action = 0
            else:
                raise ValueError(
                    f"ERROR: Invalid flow supervision type: {self.flow_supervision}"
                )

            point_loss = (
                self.action_weight * point_loss_action
                + self.anchor_weight * point_loss_anchor
            )
            dense_loss = (
                self.action_weight * dense_loss_action
                + self.anchor_weight * dense_loss_anchor
            )
            smoothness_loss = (
                self.action_weight * smoothness_loss_action
                + self.anchor_weight * smoothness_loss_anchor
            )

            loss = (
                self.displace_weight * point_loss
                + self.smoothness_weight * smoothness_loss
                + self.consistency_weight * dense_loss
            )

            log_values[loss_prefix + "point_loss"] = self.displace_weight * point_loss
            log_values[loss_prefix + "smoothness_loss"] = (
                self.smoothness_weight * smoothness_loss
            )
            log_values[loss_prefix + "dense_loss"] = (
                self.consistency_weight * dense_loss
            )

            # Calculate error metrics compared to the demo
            # TODO: always calculate these metrics, rename from 2rack to something general
            if not self.min_err_across_racks_debug:
                pass
            else:
                action_center = None
                anchor_center = None
                if (
                    self.decoder_type in ["flow", "point"]
                    and self.flow_frame == "aligned"
                ):
                    action_center = translate_to_action_center
                    anchor_center = translate_to_anchor_center

                error_R_mean, error_t_mean = get_2rack_errors(
                    pred_T_action=pred_T_action,
                    T0=T0,
                    T1=T1,
                    mode=self.error_mode_2rack,
                    T_aug_list=T_aug_list,
                    action_center=action_center,
                    anchor_center=anchor_center,
                )

                log_values[loss_prefix + "error_R_mean"] = error_R_mean
                log_values[loss_prefix + "error_t_mean"] = error_t_mean
                log_values[loss_prefix + "rotation_loss"] = (
                    self.rotation_weight * error_R_mean
                )

                if self.compute_rpdiff_min_errors and not self.training:
                    coverage_error_t_mean, coverage_error_R_mean = (
                        get_rpdiff_min_errors(
                            pred_T_action,
                            batch,
                            rpdiff_descriptions_path=self.rpdiff_descriptions_path,
                        )
                    )

                    log_values[loss_prefix + "wta_error_t_mean"] = coverage_error_t_mean
                    log_values[loss_prefix + "wta_error_R_mean"] = coverage_error_R_mean

            total_loss += loss

        # Get average loss over all samples
        loss = total_loss / n_samples

        log_values[loss_prefix + "R0_mean"] = R0_mean
        log_values[loss_prefix + "R0_max"] = R0_max
        log_values[loss_prefix + "R0_min"] = R0_min
        log_values[loss_prefix + "R1_mean"] = R1_mean
        log_values[loss_prefix + "R1_max"] = R1_max
        log_values[loss_prefix + "R1_min"] = R1_min
        log_values[loss_prefix + "t0_mean"] = t0_mean
        log_values[loss_prefix + "t0_max"] = t0_max
        log_values[loss_prefix + "t0_min"] = t0_min
        log_values[loss_prefix + "t1_mean"] = t1_mean
        log_values[loss_prefix + "t1_max"] = t1_max
        log_values[loss_prefix + "t1_min"] = t1_min

        # Compute regularization losses only for the first model output
        # Discrete latent prior regularization loss
        if self.model.conditioning in ["uniform_prior_pos_delta_l2norm"]:
            # Apply uniform prior to the goal embedding
            goal_emb = model_outputs[0]["goal_emb"]
            uniform = (
                (
                    torch.ones((goal_emb.shape[0], goal_emb.shape[1], N))
                    / goal_emb.shape[-1]
                )
                .cuda()
                .detach()
            )
            action_kl = F.kl_div(
                F.log_softmax(uniform, dim=-1),
                F.log_softmax(goal_emb[:, :, :N], dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            anchor_kl = F.kl_div(
                F.log_softmax(uniform, dim=-1),
                F.log_softmax(goal_emb[:, :, N:], dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            vae_reg_loss = action_kl + anchor_kl
            loss += self.vae_reg_loss_weight * vae_reg_loss
            log_values[loss_prefix + "vae_reg_loss"] = (
                self.vae_reg_loss_weight * vae_reg_loss
            )
        elif self.model.conditioning in ["distance_prior_pos_delta_l2norm"]:
            # Apply a prior related to the distance from selected action/anchor conditioning points
            goal_emb = model_outputs[0]["goal_emb"]

            action_distance_prior = model_outputs[0]["dense_trans_pt_action"]
            anchor_distance_prior = model_outputs[0]["dense_trans_pt_anchor"]

            # Map to [0,1]
            action_distance_prior = (
                action_distance_prior - action_distance_prior.min()
            ) / (action_distance_prior.max() - action_distance_prior.min())
            anchor_distance_prior = (
                anchor_distance_prior - anchor_distance_prior.min()
            ) / (anchor_distance_prior.max() - anchor_distance_prior.min())

            # Give higher probability to closer points
            action_distance_prior = (1 - action_distance_prior) ** 2
            anchor_distance_prior = (1 - anchor_distance_prior) ** 2

            action_kl = F.kl_div(
                F.log_softmax(action_distance_prior, dim=-1),
                F.log_softmax(goal_emb[:, :, :N], dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            anchor_kl = F.kl_div(
                F.log_softmax(anchor_distance_prior, dim=-1),
                F.log_softmax(goal_emb[:, :, N:], dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            vae_reg_loss = action_kl + anchor_kl
            loss += self.vae_reg_loss_weight * vae_reg_loss
            log_values[loss_prefix + "vae_reg_loss"] = (
                self.vae_reg_loss_weight * vae_reg_loss
            )

        # If using a continuous latent, apply a VAE regularization loss
        if heads is not None or self.model.conditioning in [
            "hybrid_pos_delta_l2norm",
            "hybrid_pos_delta_l2norm_internalcond",
            "hybrid_pos_delta_l2norm_global",
            "hybrid_pos_delta_l2norm_global_internalcond",
        ]:

            def vae_regularization_loss(mu, log_var):
                # From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/cvae.py#LL144C9-L144C105
                return torch.mean(
                    -0.5
                    * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1).mean(dim=-1),
                    dim=0,
                )

            # Regularize the global latent z to be standard normal
            if self.model.conditioning in [
                "latent_z",
                "latent_z_1pred",
                "latent_z_1pred_10d",
                "latent_z_linear",
                "latent_z_linear_internalcond",
            ]:
                vae_reg_loss = vae_regularization_loss(
                    heads["goal_emb_mu"], heads["goal_emb_logvar"]
                )
                vae_reg_loss = torch.nan_to_num(vae_reg_loss)

                loss += self.vae_reg_loss_weight * vae_reg_loss
                log_values[loss_prefix + "vae_reg_loss"] = (
                    self.vae_reg_loss_weight * vae_reg_loss
                )

            # Regularize the per-point latents to be standard normal
            elif self.model.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
            ]:
                if self.hybrid_cond_regularize_all:
                    # Regularize all per-point latents to be standard normal, not just the selected ones
                    goal_emb_mu_action = model_outputs[0]["goal_emb_mu_action"]
                    goal_emb_logvar_action = model_outputs[0]["goal_emb_logvar_action"]
                    goal_emb_mu_anchor = model_outputs[0]["goal_emb_mu_anchor"]
                    goal_emb_logvar_anchor = model_outputs[0]["goal_emb_logvar_anchor"]

                    action_vae_reg_loss = vae_regularization_loss(
                        goal_emb_mu_action, goal_emb_logvar_action
                    )
                    action_vae_reg_loss = torch.nan_to_num(action_vae_reg_loss)

                    anchor_vae_reg_loss = vae_regularization_loss(
                        goal_emb_mu_anchor, goal_emb_logvar_anchor
                    )
                    anchor_vae_reg_loss = torch.nan_to_num(anchor_vae_reg_loss)
                else:
                    # Regularize only the selected per-point latents to be standard normal
                    action_mu = model_outputs[0]["action_mu"]
                    action_logvar = model_outputs[0]["action_logvar"]
                    anchor_mu = model_outputs[0]["anchor_mu"]
                    anchor_logvar = model_outputs[0]["anchor_logvar"]

                    action_vae_reg_loss = vae_regularization_loss(
                        action_mu, action_logvar
                    )
                    action_vae_reg_loss = torch.nan_to_num(action_vae_reg_loss)

                    anchor_vae_reg_loss = vae_regularization_loss(
                        anchor_mu, anchor_logvar
                    )
                    anchor_vae_reg_loss = torch.nan_to_num(anchor_vae_reg_loss)

                loss += self.vae_reg_loss_weight * (
                    action_vae_reg_loss + anchor_vae_reg_loss
                )
                log_values[loss_prefix + "action_vae_reg_loss"] = (
                    self.vae_reg_loss_weight * action_vae_reg_loss
                )
                log_values[loss_prefix + "anchor_vae_reg_loss"] = (
                    self.vae_reg_loss_weight * anchor_vae_reg_loss
                )
                log_values[loss_prefix + "vae_reg_loss"] = self.vae_reg_loss_weight * (
                    action_vae_reg_loss + anchor_vae_reg_loss
                )

            # Regularize the global latent z to be standard normal
            elif self.model.conditioning in [
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                vae_reg_loss = vae_regularization_loss(
                    model_outputs[0]["goal_emb_mu"], model_outputs[0]["goal_emb_logvar"]
                )
                vae_reg_loss = torch.nan_to_num(vae_reg_loss)

                loss += self.vae_reg_loss_weight * vae_reg_loss
                log_values[loss_prefix + "vae_reg_loss"] = (
                    self.vae_reg_loss_weight * vae_reg_loss
                )
            else:
                raise ValueError(
                    "ERROR: Why is there a non-None heads variable passed in when the model isn't even a latent_z model?"
                )

        # Apply a spatial distance regularization loss on the selected action/anchor conditioning points
        if self.spatial_distance_regularization_weight > 0:
            if self.model.conditioning in [
                "pos_delta_l2norm",
                "uniform_prior_pos_delta_l2norm",
                "distance_prior_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_global",
            ]:
                selected_anchor_point_idx = model_outputs[0]["trans_sample_anchor"]
                _, selected_anchor_point = self.model.get_dense_translation_point(
                    points_anchor.permute(0, 2, 1)[:, :3],
                    selected_anchor_point_idx,
                    conditioning=(
                        self.model.conditioning
                        if re.match(r"^hybrid_.*$", self.model.conditioning) is None
                        else "pos_delta_l2norm"
                    ),
                )

                # Compare the demo selected anchor point to the transformed selected anchor point
                if self.spatial_distance_regularization_type == "pred":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        pred_points_action.permute(0, 2, 1)[:, :3],
                        selected_action_point_idx,
                        conditioning=(
                            self.model.conditioning
                            if re.match(r"^hybrid_.*$", self.model.conditioning) is None
                            else "pos_delta_l2norm"
                        ),
                    )
                # Compare the demo selected anchor point to the transformed selected anchor point, but don't backpropagate through TAXPose
                elif self.spatial_distance_regularization_type == "pred_sg":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        pred_points_action.permute(0, 2, 1)[:, :3].detach(),
                        selected_action_point_idx,
                        conditioning=(
                            self.model.conditioning
                            if re.match(r"^hybrid_.*$", self.model.conditioning) is None
                            else "pos_delta_l2norm"
                        ),
                    )
                # Compare the demo selected anchor point to the demo selected action point
                elif self.spatial_distance_regularization_type == "demo":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        points_action.permute(0, 2, 1)[:, :3],
                        selected_action_point_idx,
                        conditioning=(
                            self.model.conditioning
                            if re.match(r"^hybrid_.*$", self.model.conditioning) is None
                            else "pos_delta_l2norm"
                        ),
                    )
                else:
                    raise ValueError(
                        "ERROR: Invalid spatial distance regularization type"
                    )

                dist = (
                    (selected_action_point - selected_anchor_point)
                    .pow(2)
                    .sum(-1)
                    .sqrt()
                    .mean()
                )

            # Do the same as above, but learn a distance vector offset for the selected anchor point
            elif self.model.conditioning in [
                "pos_delta_l2norm_dist_vec",
                "uniform_prior_pos_delta_l2norm_dist_vec",
                "distance_prior_pos_delta_l2norm_dist_vec",
            ]:
                selected_anchor_point_idx = model_outputs[0]["trans_sample_anchor"]
                _, selected_anchor_point = self.model.get_dense_translation_point(
                    points_anchor.permute(0, 2, 1)[:, :3],
                    selected_anchor_point_idx,
                    conditioning=self.model.conditioning,
                )

                dist_vec = model_outputs[0]["dist_vec"]
                selected_anchor_dist_vec = (
                    dist_vec[:, :, N:] * selected_anchor_point_idx[:, None, :]
                ).sum(axis=-1)

                # Compare the demo selected anchor point to the transformed selected anchor point
                if self.spatial_distance_regularization_type == "pred":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        pred_points_action.permute(0, 2, 1)[:, :3],
                        selected_action_point_idx,
                        conditioning=self.model.conditioning,
                    )
                # Compare the demo selected anchor point to the transformed selected anchor point, but don't backpropagate through TAXPose
                elif self.spatial_distance_regularization_type == "pred_sg":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        pred_points_action.permute(0, 2, 1)[:, :3].detach(),
                        selected_action_point_idx,
                        conditioning=self.model.conditioning,
                    )
                # Compare the demo selected anchor point to the demo selected action point
                elif self.spatial_distance_regularization_type == "demo":
                    selected_action_point_idx = model_outputs[0]["trans_sample_action"]
                    _, selected_action_point = self.model.get_dense_translation_point(
                        points_action.permute(0, 2, 1)[:, :3],
                        selected_action_point_idx,
                        conditioning=self.model.conditioning,
                    )
                else:
                    raise ValueError(
                        "ERROR: Invalid spatial distance regularization type"
                    )

                dist = (
                    (
                        selected_action_point
                        - (selected_anchor_point + selected_anchor_dist_vec)
                    )
                    .pow(2)
                    .sum(-1)
                    .sqrt()
                    .mean()
                )

            # Do the same as above, but learn a distance scalar offset for the distance
            elif self.model.conditioning in [
                "pos_delta_l2norm_dist_scalar",
                "uniform_prior_pos_delta_l2norm_dist_scalar",
                "distance_prior_pos_delta_l2norm_dist_scalar",
            ]:
                raise NotImplementedError("TODO: Implement this")

            else:
                raise ValueError(
                    "ERROR: Invalid conditioning type for spatial distance regularization"
                )

            spatial_distance_loss = self.spatial_distance_regularization_weight * dist
            log_values[loss_prefix + "spatial_distance_regularization"] = dist
            log_values[loss_prefix + "spatial_distance_regularization_loss"] = (
                spatial_distance_loss
            )
            loss += spatial_distance_loss

        return loss, log_values

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            if self.sigmoid_on:
                pred_w = torch.sigmoid(x[:, :, 3])
            else:
                pred_w = x[:, :, 3]
        else:
            pred_w = None
        return pred_flow, pred_w

    def module_step(self, batch, batch_idx, log_prefix=""):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        points_onetrans_action = (
            batch["points_action_onetrans"]
            if "points_action_onetrans" in batch
            else batch["points_action"]
        )
        points_onetrans_anchor = (
            batch["points_anchor_onetrans"]
            if "points_anchor_onetrans" in batch
            else batch["points_anchor"]
        )

        # Model forward pass
        model_outputs = self.model(
            points_trans_action,
            points_trans_anchor,
            points_onetrans_action,
            points_onetrans_anchor,
            n_samples=self.n_samples,
        )

        # Extract from the model outputs
        # TODO only pass in points_anchor and points_action if the model is training
        if self.model.conditioning not in [
            "latent_z",
            "latent_z_1pred",
            "latent_z_1pred_10d",
            "latent_z_linear",
            "latent_z_linear_internalcond",
        ]:
            heads = [None for _ in model_outputs]
        else:
            heads = [
                {
                    "goal_emb_mu": model_output["goal_emb_mu"],
                    "goal_emb_logvar": model_output["goal_emb_logvar"],
                }
                for model_output in model_outputs
            ]

        # Compute the p(z|Y) losses
        log_values = {}
        loss, log_values = self.compute_loss(
            model_outputs,
            batch,
            log_values=log_values,
            loss_prefix=log_prefix,
            heads_list=heads,
        )

        # Auxiliary Transformation InfoNCE loss
        n_samples = len(model_outputs)
        aux_losses = 0
        for model_output in model_outputs:
            if self.pzY_taxpose_infonce_loss_weight > 0:
                action_center = None
                anchor_center = None
                if self.model.taxpose_centering == "z":
                    action_center = model_output["trans_pt_action"][:, :, None]
                    anchor_center = model_output["trans_pt_anchor"][:, :, None]

                action_infonce_loss, action_infonce_log_values = compute_infonce_loss(
                    model=self.model.tax_pose.emb_nn_action,
                    points=model_output["action_points_and_cond"]
                    .permute(0, 2, 1)
                    .detach(),
                    center=(
                        action_center.permute(0, 2, 1).detach()
                        if action_center is not None
                        else None
                    ),
                    log_prefix=log_prefix + "action_",
                )
                anchor_infonce_loss, anchor_infonce_log_values = compute_infonce_loss(
                    model=self.model.tax_pose.emb_nn_anchor,
                    points=model_output["anchor_points_and_cond"]
                    .permute(0, 2, 1)
                    .detach(),
                    center=(
                        anchor_center.permute(0, 2, 1).detach()
                        if anchor_center is not None
                        else None
                    ),
                    log_prefix=log_prefix + "anchor_",
                )

                aux_losses += self.pzY_taxpose_infonce_loss_weight * (
                    action_infonce_loss + anchor_infonce_loss
                )
                log_values = {
                    **log_values,
                    **action_infonce_log_values,
                    **anchor_infonce_log_values,
                }

            # Auxiliary Occlusion InfoNCE loss
            if self.pzY_taxpose_occ_infonce_loss_weight > 0:
                action_center = None
                anchor_center = None
                if self.model.taxpose_centering == "z":
                    action_center = model_output["trans_pt_action"][:, :, None]
                    anchor_center = model_output["trans_pt_anchor"][:, :, None]

                action_occ_infonce_loss, action_occ_infonce_log_values = (
                    compute_occlusion_infonce_loss(
                        model=self.model.tax_pose.emb_nn_action,
                        points=model_output["action_points_and_cond"]
                        .permute(0, 2, 1)
                        .detach(),
                        center=(
                            action_center.permute(0, 2, 1).detach()
                            if action_center is not None
                            else None
                        ),
                        log_prefix=log_prefix + "action_",
                    )
                )
                anchor_occ_infonce_loss, anchor_occ_infonce_log_values = (
                    compute_occlusion_infonce_loss(
                        model=self.model.tax_pose.emb_nn_anchor,
                        points=model_output["anchor_points_and_cond"]
                        .permute(0, 2, 1)
                        .detach(),
                        center=(
                            anchor_center.permute(0, 2, 1).detach()
                            if anchor_center is not None
                            else None
                        ),
                        log_prefix=log_prefix + "anchor_",
                    )
                )

                aux_losses += self.pzY_taxpose_occ_infonce_loss_weight * (
                    action_occ_infonce_loss + anchor_occ_infonce_loss
                )
                log_values = {
                    **log_values,
                    **action_occ_infonce_log_values,
                    **anchor_occ_infonce_log_values,
                }

        loss += aux_losses / n_samples

        # Debugging, plot inference errors when sampling from known prior, and get errors when sampling more than once
        if self.get_sample_errors:
            with torch.no_grad():

                def get_inference_error(log_values, batch, loss_prefix):
                    T0 = Transform3d(matrix=batch["T0"])
                    T1 = Transform3d(matrix=batch["T1"])
                    T_aug_list = (
                        [Transform3d(matrix=T_aug) for T_aug in batch["T_aug_list"]]
                        if "T_aug_list" in batch
                        else None
                    )

                    if self.model.conditioning not in [
                        "uniform_prior_pos_delta_l2norm",
                        "latent_z",
                        "latent_z_1pred",
                        "latent_z_1pred_10d",
                        "latent_z_linear",
                        "latent_z_linear_internalcond",
                    ]:
                        model_outputs = self.model(
                            points_trans_action,
                            points_trans_anchor,
                            points_onetrans_action,
                            points_onetrans_anchor,
                            mode="forward",
                            n_samples=1,
                        )
                    else:
                        model_outputs = self.model(
                            points_trans_action,
                            points_trans_anchor,
                            points_onetrans_action,
                            points_onetrans_anchor,
                            mode="inference",
                            n_samples=1,
                        )

                    x_action = model_outputs[0]["flow_action"]
                    x_anchor = model_outputs[0]["flow_anchor"]
                    goal_emb = model_outputs[0]["goal_emb"]

                    # If we've applied some sampling, we need to extract the predictions too...
                    if "sampled_ixs_action" in model_outputs[0]:
                        ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(
                            -1
                        )
                        sampled_points_trans_action = torch.take_along_dim(
                            points_trans_action, ixs_action, dim=1
                        )
                    else:
                        sampled_points_trans_action = points_trans_action

                    if "sampled_ixs_anchor" in model_outputs[0]:
                        ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(
                            -1
                        )
                        sampled_points_trans_anchor = torch.take_along_dim(
                            points_trans_anchor, ixs_anchor, dim=1
                        )
                    else:
                        sampled_points_trans_anchor = points_trans_anchor

                    pred_flow_action, pred_w_action = self.extract_flow_and_weight(
                        x_action
                    )
                    pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
                        x_anchor
                    )

                    del x_action, x_anchor, goal_emb

                    pred_T_action = dualflow2pose(
                        xyz_src=sampled_points_trans_action,
                        xyz_tgt=sampled_points_trans_anchor,
                        flow_src=pred_flow_action,
                        flow_tgt=pred_flow_anchor,
                        weights_src=pred_w_action,
                        weights_tgt=pred_w_anchor,
                        return_transform3d=True,
                        normalization_scehme=self.weight_normalize,
                        temperature=self.softmax_temperature,
                    )

                    if not self.min_err_across_racks_debug:
                        # don't print rotation/translation error metrics to the logs
                        pass
                        # error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
                        #     T1).compose(pred_T_action.inverse()))

                        # error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
                        #     T1).compose(pred_T_action.inverse()))
                    else:
                        error_R_mean, error_t_mean = get_2rack_errors(
                            pred_T_action,
                            T0,
                            T1,
                            mode=self.error_mode_2rack,
                            T_aug_list=T_aug_list,
                        )
                        log_values[loss_prefix + "sample_error_R_mean"] = error_R_mean
                        log_values[loss_prefix + "sample_error_t_mean"] = error_t_mean

                get_inference_error(log_values, batch, loss_prefix=log_prefix)

                if self.get_errors_across_samples:
                    T0 = Transform3d(matrix=batch["T0"])
                    T1 = Transform3d(matrix=batch["T1"])
                    T_aug_list = (
                        [Transform3d(matrix=T_aug) for T_aug in batch["T_aug_list"]]
                        if "T_aug_list" in batch
                        else None
                    )

                    if self.use_debug_sampling_methods:
                        sampling_settings = [("random", 1), ("top_n", 3)]
                    else:
                        sampling_settings = [("gumbel", self.n_samples)]

                    all_model_outputs = []
                    for sampling_setting in sampling_settings:
                        if self.model.conditioning not in [
                            "uniform_prior_pos_delta_l2norm",
                            "latent_z",
                            "latent_z_1pred",
                            "latent_z_1pred_10d",
                            "latent_z_linear",
                            "latent_z_linear_internalcond",
                        ]:
                            model_outputs = self.model(
                                points_trans_action,
                                points_trans_anchor,
                                points_onetrans_action,
                                points_onetrans_anchor,
                                mode="forward",
                                sampling_method=sampling_setting[0],
                                n_samples=sampling_setting[1],
                            )
                        else:
                            model_outputs = self.model(
                                points_trans_action,
                                points_trans_anchor,
                                points_onetrans_action,
                                points_onetrans_anchor,
                                mode="inference",
                                sampling_method=sampling_setting[0],
                                n_samples=sampling_setting[1],
                            )
                        all_model_outputs.append(model_outputs)

                    all_predictions = []
                    for i in range(len(all_model_outputs)):
                        setting_predictions = []
                        for j in range(len(all_model_outputs[i])):
                            x_action = all_model_outputs[i][j]["flow_action"]
                            x_anchor = all_model_outputs[i][j]["flow_anchor"]
                            goal_emb = all_model_outputs[i][j]["goal_emb"]

                            # If we've applied some sampling, we need to extract the predictions too...
                            if "sampled_ixs_action" in model_outputs[0]:
                                ixs_action = model_outputs[0][
                                    "sampled_ixs_action"
                                ].unsqueeze(-1)
                                sampled_points_trans_action = torch.take_along_dim(
                                    points_trans_action, ixs_action, dim=1
                                )
                            else:
                                sampled_points_trans_action = points_trans_action

                            if "sampled_ixs_anchor" in model_outputs[0]:
                                ixs_anchor = model_outputs[0][
                                    "sampled_ixs_anchor"
                                ].unsqueeze(-1)
                                sampled_points_trans_anchor = torch.take_along_dim(
                                    points_trans_anchor, ixs_anchor, dim=1
                                )
                            else:
                                sampled_points_trans_anchor = points_trans_anchor

                            # Get flow/weights
                            pred_flow_action, pred_w_action = (
                                self.extract_flow_and_weight(x_action)
                            )
                            pred_flow_anchor, pred_w_anchor = (
                                self.extract_flow_and_weight(x_anchor)
                            )

                            # Get predicted transform
                            pred_T_action = dualflow2pose(
                                xyz_src=sampled_points_trans_action,
                                xyz_tgt=sampled_points_trans_anchor,
                                flow_src=pred_flow_action,
                                flow_tgt=pred_flow_anchor,
                                weights_src=pred_w_action,
                                weights_tgt=pred_w_anchor,
                                return_transform3d=True,
                                normalization_scehme=self.weight_normalize,
                                temperature=self.softmax_temperature,
                            )

                            setting_predictions.append(pred_T_action)
                        all_predictions.append(setting_predictions)

                    for i in range(len(all_predictions)):
                        (
                            error_R_maxs,
                            error_R_mins,
                            error_R_means,
                            error_t_maxs,
                            error_t_mins,
                            error_t_means,
                        ) = get_all_sample_errors(
                            all_predictions[i],
                            T0,
                            T1,
                            mode=self.error_mode_2rack,
                            T_aug_list=T_aug_list,
                        )
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_mean"
                        ] = torch.mean(torch.Tensor(error_R_means))
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_mean"
                        ] = torch.mean(torch.Tensor(error_t_means))
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_max"
                        ] = torch.max(torch.Tensor(error_R_maxs))
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_max"
                        ] = torch.max(torch.Tensor(error_t_maxs))
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_min"
                        ] = torch.min(torch.Tensor(error_R_mins))
                        log_values[
                            log_prefix
                            + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_min"
                        ] = torch.min(torch.Tensor(error_t_mins))

        return loss, log_values

    def visualize_results(self, batch, batch_idx, log_prefix=""):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        points_onetrans_action = (
            batch["points_action_onetrans"]
            if "points_action_onetrans" in batch
            else batch["points_action"]
        )
        points_onetrans_anchor = (
            batch["points_anchor_onetrans"]
            if "points_anchor_onetrans" in batch
            else batch["points_anchor"]
        )

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        # Model forward pass
        model_outputs = self.model(
            points_trans_action,
            points_trans_anchor,
            points_onetrans_action,
            points_onetrans_anchor,
            n_samples=1,
        )
        # Extract from the model outputs
        goal_emb = model_outputs[0]["goal_emb"]

        # Only use XYZ
        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]
        points_onetrans_action = points_onetrans_action[:, :, :3]
        points_onetrans_anchor = points_onetrans_anchor[:, :, :3]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_outputs[0]:
            ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_action = torch.take_along_dim(
                points_action, ixs_action, dim=1
            )
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_action = points_action
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_outputs[0]:
            ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_anchor = torch.take_along_dim(
                points_anchor, ixs_anchor, dim=1
            )
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_anchor = points_anchor
            sampled_points_trans_anchor = points_trans_anchor

        # Get predicted transform
        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_outputs[0]["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            x_anchor = model_outputs[0]["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action,
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_outputs[0]["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_outputs[0]["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(
                f"ERROR: Invalid flow supervision type: {self.flow_supervision}"
            )

        if self.decoder_type == "taxpose":
            pred_points_action = pred_T_action.transform_points(points_trans_action)
        elif self.decoder_type in ["flow", "point"] and self.flow_frame == "aligned":
            action_center = model_outputs[0]["trans_pt_action"][:, :, None]
            anchor_center = model_outputs[0]["trans_pt_anchor"][:, :, None]

            points_trans_action_in_aligned_frame = (
                points_trans_action - action_center.permute(0, 2, 1)
            )
            points_action_pred_in_aligned_frame = pred_T_action.transform_points(
                points_trans_action_in_aligned_frame
            )
            pred_points_action = (
                points_action_pred_in_aligned_frame + anchor_center.permute(0, 2, 1)
            )
        else:
            raise ValueError(f"ERROR: Invalid decoder type: {self.decoder_type}")

        # Logging results
        res_images = {}

        demo_points_tensors = [points_onetrans_action[0], points_onetrans_anchor[0]]
        demo_points_colors = ["blue", "red"]
        if "points_action_aug_trans" in batch:
            demo_points_tensors.append(batch["points_action_aug_trans"][0, :, :3])
            demo_points_colors.append("yellow")
        demo_points = get_color(
            tensor_list=demo_points_tensors, color_list=demo_points_colors
        )
        res_images[log_prefix + "demo_points"] = wandb.Object3D(demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "action_transformed_action"] = wandb.Object3D(
            action_transformed_action
        )

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "anchor_transformed_anchor"] = wandb.Object3D(
            anchor_transformed_anchor
        )

        # transformed_input_points = get_color(tensor_list=[
        #     points_trans_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        # res_images[log_prefix+'transformed_input_points'] = wandb.Object3D(
        #     transformed_input_points)

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "demo_points_apply_action_transform"] = wandb.Object3D(
            demo_points_apply_action_transform
        )

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "apply_action_transform_demo_comparable"] = (
            wandb.Object3D(apply_action_transform_demo_comparable)
        )

        predicted_vs_gt_transform_tensors = [
            T1.inverse().transform_points(pred_points_action)[0],
            points_action[0],
            T1.inverse().transform_points(points_trans_anchor)[0],
        ]
        predicted_vs_gt_transform_colors = ["blue", "green", "red"]
        if "points_action_aug_trans" in batch:
            predicted_vs_gt_transform_tensors.append(
                batch["points_action_aug_trans"][0, :, :3]
            )
            predicted_vs_gt_transform_colors.append("yellow")
        predicted_vs_gt_transform_applied = get_color(
            tensor_list=predicted_vs_gt_transform_tensors,
            color_list=predicted_vs_gt_transform_colors,
        )
        res_images[log_prefix + "predicted_vs_gt_transform_applied"] = wandb.Object3D(
            predicted_vs_gt_transform_applied
        )

        apply_predicted_transform = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=[
                "blue",
                "orange",
                "red",
            ],
        )
        res_images[log_prefix + "apply_predicted_transform"] = wandb.Object3D(
            apply_predicted_transform
        )

        # loss_points_action = get_color(
        #     tensor_list=[points_action_target[0], pred_points_action[0]], color_list=['green', 'red'])
        # res_images[log_prefix+'loss_points_action'] = wandb.Object3D(
        #     loss_points_action)

        pred_w_points_list = []
        pred_w_colors_list = []
        if self.flow_supervision in ["both", "action2anchor"]:
            pred_w_points_list.append(sampled_points_action[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_action[0]))
        if self.flow_supervision in ["both", "anchor2action"]:
            pred_w_points_list.append(sampled_points_anchor[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_anchor[0]))

        pred_w_on_objects = np.concatenate(
            [
                torch.cat(pred_w_points_list, dim=0).cpu().numpy(),
                np.concatenate(pred_w_colors_list, axis=0),
            ],
            axis=-1,
        )

        res_images[log_prefix + "pred_w"] = wandb.Object3D(
            pred_w_on_objects, markerSize=1000
        )

        # This visualization only applies to methods that have discrete per-point latents
        if self.model.conditioning not in [
            "latent_z_linear",
            "latent_z_linear_internalcond",
        ]:
            if self.model.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
            ]:
                goal_emb = goal_emb[:, :1, :]

            # Plot goal embeddings on objects
            goal_emb_norm_action = (
                F.softmax(goal_emb[0, :, : points_action.shape[1]], dim=-1)
                .detach()
                .cpu()
            )
            goal_emb_norm_anchor = (
                F.softmax(goal_emb[0, :, points_action.shape[1] :], dim=-1)
                .detach()
                .cpu()
            )
            colors_action = color_gradient(goal_emb_norm_action[0])
            colors_anchor = color_gradient(goal_emb_norm_anchor[0])
            goal_emb_on_objects = np.concatenate(
                [
                    torch.cat(
                        [points_action[0].detach(), points_anchor[0].detach()], dim=0
                    )
                    .cpu()
                    .numpy(),
                    np.concatenate([colors_action, colors_anchor], axis=0),
                ],
                axis=-1,
            )
            res_images[log_prefix + "goal_emb"] = wandb.Object3D(goal_emb_on_objects)

            # Plot the p(z|Y) embeddings on the objects
            if self.plot_encoder_distribution:
                # Get embeddings for p(z|Y2)
                if "points_action_aug_trans" in batch:
                    points_action_aug_trans = batch["points_action_aug_trans"]
                    points_trans_action = batch["points_action_trans"]
                    points_trans_anchor = batch["points_anchor_trans"]
                    points_action = batch["points_action"]
                    points_anchor = batch["points_anchor"]
                    points_onetrans_action = (
                        batch["points_action_onetrans"]
                        if "points_action_onetrans" in batch
                        else batch["points_action"]
                    )
                    points_onetrans_anchor = (
                        batch["points_anchor_onetrans"]
                        if "points_anchor_onetrans" in batch
                        else batch["points_anchor"]
                    )

                    assert (
                        points_action_aug_trans.shape[1]
                        == points_onetrans_anchor.shape[1]
                    ), f"ERROR: Augmented action points have different number of points than the anchor points. Are you trying to visualize with more than 1 distractor?"

                    model_no_cond_x_outputs = self.model(
                        points_trans_action,
                        points_trans_anchor,
                        points_action_aug_trans,
                        points_onetrans_anchor,
                        n_samples=1,
                    )

                    pzY2_emb = model_no_cond_x_outputs[0]["goal_emb"]
                    if self.model.conditioning in [
                        "hybrid_pos_delta_l2norm",
                        "hybrid_pos_delta_l2norm_internalcond",
                    ]:
                        pzY2_emb = pzY2_emb[:, :1, :]

                    pzY2_action_emb = pzY2_emb[0, :, : points_action.shape[1]]
                    pzY2_anchor_emb = pzY2_emb[0, :, points_action.shape[1] :]
                    pzY2_action_dist = (
                        F.softmax(pzY2_action_emb, dim=-1).detach().cpu().numpy()[0]
                    )
                    pzY2_anchor_dist = (
                        F.softmax(pzY2_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                    )

                    pzY2_actions_probs = np.array([prob for prob in pzY2_action_dist])
                    pzY2_anchor_probs = np.array([prob for prob in pzY2_anchor_dist])

                pzY1_action_emb = goal_emb[0, :, : points_action.shape[1]]
                pzY1_anchor_emb = goal_emb[0, :, points_action.shape[1] :]
                pzY1_action_dist = (
                    F.softmax(pzY1_action_emb, dim=-1).detach().cpu().numpy()[0]
                )
                pzY1_anchor_dist = (
                    F.softmax(pzY1_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                )

                x_vals = np.arange(pzY1_action_dist.shape[0])
                pzY1_actions_probs = np.array([prob for prob in pzY1_action_dist])
                pzY1_anchor_probs = np.array([prob for prob in pzY1_anchor_dist])

                layout_margin = go.layout.Margin(
                    l=50,  # left margin
                    r=120,  # right margin
                    b=50,  # bottom margin
                    t=50,  # top margin
                    autoexpand=False,
                )

                # Plot action distributions
                action_max_prob = np.max(pzY1_actions_probs)

                pzY_action_data = [
                    go.Bar(
                        name="pzY1",
                        x=x_vals,
                        y=pzY1_actions_probs,
                        width=1,
                        marker_color="blue",
                        opacity=0.5,
                        showlegend=True,
                    )
                ]
                if "points_action_aug_trans" in batch:
                    pzY_action_data.append(
                        go.Bar(
                            name="pzY2",
                            x=x_vals,
                            y=pzY2_actions_probs,
                            width=1,
                            marker_color="red",
                            opacity=0.5,
                            showlegend=True,
                        )
                    )
                    action_max_prob = max(action_max_prob, np.max(pzY2_actions_probs))

                pzY_action_plot = go.Figure(data=pzY_action_data)
                pzY_action_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, action_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                # Plot anchor distributions
                anchor_max_prob = np.max(pzY1_anchor_probs)

                pzY_anchor_data = [
                    go.Bar(
                        name="pzY1",
                        x=x_vals,
                        y=pzY1_anchor_probs,
                        width=1,
                        marker_color="blue",
                        opacity=0.5,
                        showlegend=True,
                    )
                ]
                if "points_action_aug_trans" in batch:
                    pzY_anchor_data.append(
                        go.Bar(
                            name="pzY2",
                            x=x_vals,
                            y=pzY2_anchor_probs,
                            width=1,
                            marker_color="red",
                            opacity=0.5,
                            showlegend=True,
                        )
                    )
                    anchor_max_prob = max(anchor_max_prob, np.max(pzY2_anchor_probs))

                pzY_anchor_plot = go.Figure(data=pzY_anchor_data)
                pzY_anchor_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, anchor_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                res_images[log_prefix + "pzY_action_distribution"] = wandb.Html(
                    plotly.io.to_html(pzY_action_plot, include_plotlyjs="cdn")
                )
                res_images[log_prefix + "pzY_anchor_distribution"] = wandb.Html(
                    plotly.io.to_html(pzY_anchor_plot, include_plotlyjs="cdn")
                )

        return res_images


class EquivarianceTrainingModule_WithPZCondX(PointCloudTrainingModule):

    def __init__(
        self,
        model_with_cond_x,
        training_module_no_cond_x,
        goal_emb_cond_x_loss_weight=1,
        joint_train_prior=False,
        freeze_residual_flow=False,
        freeze_z_embnn=False,
        freeze_embnn=False,
        n_samples=1,
        get_errors_across_samples=False,
        use_debug_sampling_methods=False,
        plot_encoder_distribution=False,
        pzX_use_pzY_z_samples=False,
        goal_emb_cond_x_loss_type="forward_kl",
        joint_infonce_loss_weight=0.0,
        spatial_distance_regularization_type="demo",
        spatial_distance_regularization_weight=0.0,
        overwrite_loss=False,
        pzX_adversarial=False,
        hybrid_cond_pzX_regularize_type=True,
        hybrid_cond_pzX_sample_latent=False,
    ):

        super().__init__(
            model=model_with_cond_x,
            lr=training_module_no_cond_x.lr,
            image_log_period=training_module_no_cond_x.image_log_period,
        )

        self.model_with_cond_x = model_with_cond_x
        self.model = self.model_with_cond_x.residflow_embnn
        self.training_module_no_cond_x = training_module_no_cond_x
        self.goal_emb_cond_x_loss_weight = goal_emb_cond_x_loss_weight
        self.goal_emb_cond_x_loss_type = goal_emb_cond_x_loss_type

        self.joint_train_prior = joint_train_prior
        self.cfg_freeze_residual_flow = freeze_residual_flow
        self.cfg_freeze_z_embnn = freeze_z_embnn
        self.cfg_freeze_embnn = freeze_embnn

        self.n_samples = n_samples
        self.get_errors_across_samples = get_errors_across_samples
        self.use_debug_sampling_methods = use_debug_sampling_methods
        self.plot_encoder_distribution = plot_encoder_distribution
        self.pzX_use_pzY_z_samples = pzX_use_pzY_z_samples

        self.joint_infonce_loss_weight = joint_infonce_loss_weight
        self.spatial_distance_regularization_type = spatial_distance_regularization_type
        self.spatial_distance_regularization_weight = (
            spatial_distance_regularization_weight
        )

        self.overwrite_loss = overwrite_loss or (
            self.cfg_freeze_embnn
            and self.cfg_freeze_z_embnn
            and self.cfg_freeze_residual_flow
        )
        self.pzX_adversarial = pzX_adversarial
        self.hybrid_cond_pzX_regularize_type = hybrid_cond_pzX_regularize_type
        self.hybrid_cond_pzX_sample_latent = hybrid_cond_pzX_sample_latent

        self.flow_supervision = self.training_module_no_cond_x.flow_supervision
        self.decoder_type = self.training_module_no_cond_x.decoder_type
        self.flow_frame = self.training_module_no_cond_x.flow_frame

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

    def get_transform(
        self,
        points_trans_action,
        points_trans_anchor,
        points_action=None,
        points_anchor=None,
        mode="forward",
        sampling_method="gumbel",
        n_samples=1,
    ):
        # mode is unused
        sample_latent = (
            self.model_with_cond_x.conditioning
            in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]
            and self.hybrid_cond_pzX_sample_latent
        )
        model_outputs = self.model_with_cond_x(
            points_trans_action,
            points_trans_anchor,
            points_action,
            points_anchor,
            sampling_method=sampling_method,
            n_samples=n_samples,
            sample_latent=sample_latent,
        )

        ans_dicts = []
        for i in range(n_samples):
            model_output = model_outputs[i]
            goal_emb = model_output["goal_emb"]
            goal_emb_cond_x = model_output["goal_emb_cond_x"]

            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]
            ans_dict = self.predict(
                model_output=model_output,
                points_trans_action=points_trans_action,
                points_trans_anchor=points_trans_anchor,
            )

            ans_dict["flow_components"] = model_output
            if self.model_with_cond_x.return_debug:
                for_debug["goal_emb"] = goal_emb
                for_debug["goal_emb_cond_x"] = goal_emb_cond_x
                ans_dict["for_debug"] = for_debug
            ans_dicts.append(ans_dict)
        return ans_dicts

    def predict(self, model_output, points_trans_action, points_trans_anchor):
        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_trans_anchor = points_trans_anchor

        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action,
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            # Get predicted transform
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(
                f"ERROR: Invalid flow supervision type: {self.flow_supervision}"
            )

        if self.decoder_type == "taxpose":
            pred_points_action = pred_T_action.transform_points(points_trans_action)
        elif self.decoder_type in ["flow", "point"] and self.flow_frame == "aligned":
            action_center = model_output["trans_pt_action"][:, :, None]
            anchor_center = model_output["trans_pt_anchor"][:, :, None]

            points_trans_action_in_aligned_frame = (
                points_trans_action - action_center.permute(0, 2, 1)
            )
            points_action_pred_in_aligned_frame = pred_T_action.transform_points(
                points_trans_action_in_aligned_frame
            )
            pred_points_action = (
                points_action_pred_in_aligned_frame + anchor_center.permute(0, 2, 1)
            )
        else:
            raise ValueError(f"ERROR: Invalid decoder type: {self.decoder_type}")

        return {
            "pred_T_action": pred_T_action,
            "pred_points_action": pred_points_action,
        }

    def compute_loss(self, model_outputs, batch, log_values={}, loss_prefix=""):
        N_action = batch["points_action"].shape[1]
        N_anchor = batch["points_anchor"].shape[1]

        # Calculate p(z|Y) losses using p(z|X) model outputs for every sample
        loss, log_values = self.training_module_no_cond_x.compute_loss(
            model_outputs, batch, log_values, loss_prefix
        )

        # Compute regularization losses only for first sample (goal embeddings should be the same across samples)
        goal_emb = model_outputs[0]["goal_emb"]
        goal_emb_cond_x = model_outputs[0]["goal_emb_cond_x"]

        # aka "if it is training time and not val time"
        if goal_emb is not None:
            B, K, D = goal_emb.shape

            # Calculate losses between p(z|Y) and p(z|X)
            if self.model_with_cond_x.conditioning not in [
                "latent_z_linear",
                "latent_z_linear_internalcond",
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                if self.goal_emb_cond_x_loss_type == "forward_kl":
                    action_kl = F.kl_div(
                        F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    anchor_kl = F.kl_div(
                        F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )

                elif self.goal_emb_cond_x_loss_type == "reverse_kl":
                    action_kl = F.kl_div(
                        F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                        F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    anchor_kl = F.kl_div(
                        F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                        F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                elif self.goal_emb_cond_x_loss_type in ["js_div", "js_div_eps0"]:
                    eps = 1e-8 if self.goal_emb_cond_x_loss_type == "js_div" else 0

                    action_kl = js_div(
                        q=F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        p=F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                        reduction="batchmean",
                        eps=eps,
                    )
                    anchor_kl = js_div(
                        q=F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        p=F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                        reduction="batchmean",
                        eps=eps,
                    )

                elif self.goal_emb_cond_x_loss_type.startswith(
                    "js_div_mod"
                ) or self.goal_emb_cond_x_loss_type.startswith("js_div_mod_eps0"):
                    eps = (
                        1e-8
                        if self.goal_emb_cond_x_loss_type.startswith("js_div_mod")
                        else 0
                    )
                    d_1 = float(self.goal_emb_cond_x_loss_type.split("_")[-2])
                    d_2 = float(self.goal_emb_cond_x_loss_type.split("_")[-1])

                    action_kl = js_div_mod(
                        q=F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        p=F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                        d_1=d_1,
                        d_2=d_2,
                        reduction="batchmean",
                        eps=eps,
                    )
                    anchor_kl = js_div_mod(
                        q=F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        p=F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                        d_1=d_1,
                        d_2=d_2,
                        reduction="batchmean",
                        eps=eps,
                    )
                elif self.goal_emb_cond_x_loss_type == "mse":
                    action_kl = F.mse_loss(
                        F.softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        F.softmax(goal_emb[:, :, :N_action], dim=-1),
                        reduction="sum",
                    )

                    anchor_kl = F.mse_loss(
                        F.softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        F.softmax(goal_emb[:, :, N_action:], dim=-1),
                        reduction="sum",
                    )

                elif self.goal_emb_cond_x_loss_type.startswith("wasserstein"):
                    p_exp = float(self.goal_emb_cond_x_loss_type.split("_")[-1])
                    action_kl = wasserstein_distance(
                        F.softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                        F.softmax(goal_emb[:, :, :N_action], dim=-1),
                        p_exp=p_exp,
                    )
                    anchor_kl = wasserstein_distance(
                        F.softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                        F.softmax(goal_emb[:, :, N_action:], dim=-1),
                        p_exp=p_exp,
                    )

                else:
                    raise ValueError(
                        f"goal_emb_cond_x_loss_type={self.goal_emb_cond_x_loss_type} not supported"
                    )

            elif self.model_with_cond_x.conditioning in [
                "latent_z_linear",
                "latent_z_linear_internalcond",
            ]:
                goal_emb_mu = model_outputs[0]["pzY_goal_emb_mu"]
                goal_emb_logvar = model_outputs[0]["pzY_goal_emb_logvar"]

                goal_emb_cond_x_mu = model_outputs[0]["goal_emb_mu"]
                goal_emb_cond_x_logvar = model_outputs[0]["goal_emb_logvar"]

                def latent_kl(mu1, logvar1, mu2, logvar2, reduction="batchmean"):
                    # KL divergence between two Gaussian distributions, KL(N(mu1, exp(logvar1)), N(mu2, exp(logvar2)))
                    kl = (
                        0.5 * (logvar2 - logvar1)
                        + (torch.exp(logvar1) + (mu1 - mu2) ** 2)
                        / (2 * torch.exp(logvar2) + 1e-8)
                        - 0.5
                    )
                    kl = torch.sum(kl, dim=1)

                    if reduction == "mean":
                        kl = torch.mean(kl)
                    elif reduction == "batchmean":
                        kl = torch.sum(kl) / kl.shape[0]
                    elif reduction == "sum":
                        kl = torch.sum(kl)
                    elif reduction == "none":
                        pass
                    else:
                        raise ValueError(f"reduction={reduction} not supported")

                    return kl

                if self.goal_emb_cond_x_loss_type == "forward_kl":
                    latent_goal_kl = latent_kl(
                        mu1=goal_emb_mu,
                        logvar1=goal_emb_logvar,
                        mu2=goal_emb_cond_x_mu,
                        logvar2=goal_emb_cond_x_logvar,
                        reduction="batchmean",
                    )

                    action_kl = latent_goal_kl
                    anchor_kl = latent_goal_kl
                elif self.goal_emb_cond_x_loss_type == "reverse_kl":
                    latent_goal_kl = latent_kl(
                        mu1=goal_emb_cond_x_mu,
                        logvar1=goal_emb_cond_x_logvar,
                        mu2=goal_emb_mu,
                        logvar2=goal_emb_logvar,
                        reduction="batchmean",
                    )

                    action_kl = latent_goal_kl
                    anchor_kl = latent_goal_kl
                elif self.goal_emb_cond_x_loss_type in ["js_div", "js_div_eps0"]:
                    eps = 1e-8 if self.goal_emb_cond_x_loss_type == "js_div" else 0

                    def compute_js_loss(
                        source_mu, source_log_var, target_mu, target_log_var
                    ):
                        # From https://discuss.pytorch.org/t/compute-js-loss-between-gaussian-distributions-parameterized-by-mu-and-log-var/130935
                        def get_prob(mu, log_var):
                            dist = torch.distributions.Normal(
                                mu, torch.exp(0.5 * log_var)
                            )
                            val = dist.sample()
                            return dist.log_prob(val).exp()

                        def kl_loss(p, q):
                            return F.kl_div(
                                p, q, reduction="batchmean", log_target=False
                            )

                        source_prob = get_prob(source_mu, source_log_var)
                        target_prob = get_prob(target_mu, target_log_var)

                        log_mean_prob = (0.5 * (source_prob + target_prob)).log()
                        js_loss = 0.5 * (
                            kl_loss(log_mean_prob, source_prob)
                            + kl_loss(log_mean_prob, target_prob)
                        )
                        return js_loss

                    n_samples = 1
                    latent_goal_kl = 0
                    for n_sample in range(n_samples):
                        latent_goal_kl += compute_js_loss(
                            source_mu=goal_emb_mu,
                            source_log_var=goal_emb_logvar,
                            target_mu=goal_emb_cond_x_mu,
                            target_log_var=goal_emb_cond_x_logvar,
                        )
                    latent_goal_kl /= n_samples

                    action_kl = latent_goal_kl
                    anchor_kl = latent_goal_kl

            elif self.model_with_cond_x.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]:
                goal_emb_discrete_action = goal_emb[:, :1, :N_action]
                goal_emb_discrete_anchor = goal_emb[:, :1, N_action:]
                goal_emb_cond_x_discrete_action = goal_emb_cond_x[:, :1, :N_action]
                goal_emb_cond_x_discrete_anchor = goal_emb_cond_x[:, :1, N_action:]

                if self.hybrid_cond_pzX_regularize_type in ["all", "selected"]:
                    goal_emb_mu_action = model_outputs[0]["pzY_goal_emb_mu_action"]
                    goal_emb_mu_anchor = model_outputs[0]["pzY_goal_emb_mu_anchor"]
                    goal_emb_logvar_action = model_outputs[0][
                        "pzY_goal_emb_logvar_action"
                    ]
                    goal_emb_logvar_anchor = model_outputs[0][
                        "pzY_goal_emb_logvar_anchor"
                    ]
                    goal_emb_selected_action_mu = model_outputs[0]["pzY_action_mu"]
                    goal_emb_selected_anchor_mu = model_outputs[0]["pzY_anchor_mu"]
                    goal_emb_selected_action_logvar = model_outputs[0][
                        "pzY_action_logvar"
                    ]
                    goal_emb_selected_anchor_logvar = model_outputs[0][
                        "pzY_anchor_logvar"
                    ]

                    goal_emb_cond_x_mu_action = model_outputs[0]["goal_emb_mu_action"]
                    goal_emb_cond_x_mu_anchor = model_outputs[0]["goal_emb_mu_anchor"]
                    goal_emb_cond_x_logvar_action = model_outputs[0][
                        "goal_emb_logvar_action"
                    ]
                    goal_emb_cond_x_logvar_anchor = model_outputs[0][
                        "goal_emb_logvar_anchor"
                    ]
                    goal_emb_cond_x_selected_action_mu = model_outputs[0]["action_mu"]
                    goal_emb_cond_x_selected_anchor_mu = model_outputs[0]["anchor_mu"]
                    goal_emb_cond_x_selected_action_logvar = model_outputs[0][
                        "action_logvar"
                    ]
                    goal_emb_cond_x_selected_anchor_logvar = model_outputs[0][
                        "anchor_logvar"
                    ]
                elif self.hybrid_cond_pzX_regularize_type in ["global"]:
                    goal_emb_mu = model_outputs[0]["pzY_goal_emb_mu"]
                    goal_emb_logvar = model_outputs[0]["pzY_goal_emb_logvar"]

                    goal_emb_cond_x_mu = model_outputs[0]["goal_emb_mu"]
                    goal_emb_cond_x_logvar = model_outputs[0]["goal_emb_logvar"]

                def latent_kl(mu1, logvar1, mu2, logvar2, reduction="batchmean"):
                    # KL divergence between two Gaussian distributions, KL(N(mu1, exp(logvar1)), N(mu2, exp(logvar2)))
                    kl = (
                        0.5 * (logvar2 - logvar1)
                        + (torch.exp(logvar1) + (mu1 - mu2) ** 2)
                        / (2 * torch.exp(logvar2) + 1e-8)
                        - 0.5
                    )
                    kl = torch.sum(kl, dim=1)

                    if reduction == "mean":
                        kl = torch.mean(kl)
                    elif reduction == "batchmean":
                        kl = torch.sum(kl) / kl.shape[0]
                    elif reduction == "sum":
                        kl = torch.sum(kl)
                    elif reduction == "none":
                        pass
                    else:
                        raise ValueError(f"reduction={reduction} not supported")

                    return kl

                discrete_action_kl = None
                discrete_anchor_kl = None
                latent_action_kl = None
                latent_anchor_kl = None
                latent_global_kl = None
                if self.goal_emb_cond_x_loss_type == "forward_kl":
                    # F.kl_div is Q then P input
                    discrete_action_kl = F.kl_div(
                        F.log_softmax(goal_emb_cond_x_discrete_action, dim=-1),
                        F.log_softmax(goal_emb_discrete_action, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    discrete_anchor_kl = F.kl_div(
                        F.log_softmax(goal_emb_cond_x_discrete_anchor, dim=-1),
                        F.log_softmax(goal_emb_discrete_anchor, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )

                    # Do forward KL for latent
                    if self.hybrid_cond_pzX_regularize_type == "all":
                        # latent_kl is P then Q input
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_mu_action,
                            logvar1=goal_emb_logvar_action,
                            mu2=goal_emb_cond_x_mu_action,
                            logvar2=goal_emb_cond_x_logvar_action,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_mu_anchor,
                            logvar1=goal_emb_logvar_anchor,
                            mu2=goal_emb_cond_x_mu_anchor,
                            logvar2=goal_emb_cond_x_logvar_anchor,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl
                    elif self.hybrid_cond_pzX_regularize_type == "selected":
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_selected_action_mu,
                            logvar1=goal_emb_selected_action_logvar,
                            mu2=goal_emb_cond_x_selected_action_mu,
                            logvar2=goal_emb_cond_x_selected_action_logvar,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_selected_anchor_mu,
                            logvar1=goal_emb_selected_anchor_logvar,
                            mu2=goal_emb_cond_x_selected_anchor_mu,
                            logvar2=goal_emb_cond_x_selected_anchor_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "global":
                        latent_global_kl = latent_kl(
                            mu1=goal_emb_mu,
                            logvar1=goal_emb_logvar,
                            mu2=goal_emb_cond_x_mu,
                            logvar2=goal_emb_cond_x_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "none":
                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl
                    else:
                        raise ValueError(
                            f"hybrid_cond_pzX_regularize_type={self.hybrid_cond_pzX_regularize_type} not supported"
                        )

                elif self.goal_emb_cond_x_loss_type == "reverse_kl":
                    discrete_action_kl = F.kl_div(
                        F.log_softmax(goal_emb_discrete_action, dim=-1),
                        F.log_softmax(goal_emb_cond_x_discrete_action, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    discrete_anchor_kl = F.kl_div(
                        F.log_softmax(goal_emb_discrete_anchor, dim=-1),
                        F.log_softmax(goal_emb_cond_x_discrete_anchor, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )

                    # Do Reverse KL for latent
                    if self.hybrid_cond_pzX_regularize_type == "all":
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu_action,
                            logvar1=goal_emb_cond_x_logvar_action,
                            mu2=goal_emb_mu_action,
                            logvar2=goal_emb_logvar_action,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu_anchor,
                            logvar1=goal_emb_cond_x_logvar_anchor,
                            mu2=goal_emb_mu_anchor,
                            logvar2=goal_emb_logvar_anchor,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl
                    elif self.hybrid_cond_pzX_regularize_type == "selected":
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_cond_x_selected_action_mu,
                            logvar1=goal_emb_cond_x_selected_action_logvar,
                            mu2=goal_emb_selected_action_mu,
                            logvar2=goal_emb_selected_action_logvar,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_cond_x_selected_anchor_mu,
                            logvar1=goal_emb_cond_x_selected_anchor_logvar,
                            mu2=goal_emb_selected_anchor_mu,
                            logvar2=goal_emb_selected_anchor_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "global":
                        latent_global_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu,
                            logvar1=goal_emb_cond_x_logvar,
                            mu2=goal_emb_mu,
                            logvar2=goal_emb_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "none":
                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl
                    else:
                        raise ValueError(
                            f"hybrid_cond_pzX_regularize_type={self.hybrid_cond_pzX_regularize_type} not supported"
                        )

                elif self.goal_emb_cond_x_loss_type in ["js_div", "js_div_eps0"]:
                    eps = 1e-8 if self.goal_emb_cond_x_loss_type == "js_div" else 0
                    discrete_action_kl = js_div(
                        q=F.log_softmax(goal_emb_cond_x_discrete_action, dim=-1),
                        p=F.log_softmax(goal_emb_discrete_action, dim=-1),
                        reduction="batchmean",
                        eps=eps,
                    )
                    discrete_anchor_kl = js_div(
                        q=F.log_softmax(goal_emb_cond_x_discrete_anchor, dim=-1),
                        p=F.log_softmax(goal_emb_discrete_anchor, dim=-1),
                        reduction="batchmean",
                        eps=eps,
                    )

                    # TODO: JSD for latent?
                    # Do Reverse KL for latent
                    if self.hybrid_cond_pzX_regularize_type == "all":
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu_action,
                            logvar1=goal_emb_cond_x_logvar_action,
                            mu2=goal_emb_mu_action,
                            logvar2=goal_emb_logvar_action,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu_anchor,
                            logvar1=goal_emb_cond_x_logvar_anchor,
                            mu2=goal_emb_mu_anchor,
                            logvar2=goal_emb_logvar_anchor,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl
                    elif self.hybrid_cond_pzX_regularize_type == "selected":
                        latent_action_kl = latent_kl(
                            mu1=goal_emb_cond_x_selected_action_mu,
                            logvar1=goal_emb_cond_x_selected_action_logvar,
                            mu2=goal_emb_selected_action_mu,
                            logvar2=goal_emb_selected_action_logvar,
                            reduction="batchmean",
                        )
                        latent_anchor_kl = latent_kl(
                            mu1=goal_emb_cond_x_selected_anchor_mu,
                            logvar1=goal_emb_cond_x_selected_anchor_logvar,
                            mu2=goal_emb_selected_anchor_mu,
                            logvar2=goal_emb_selected_anchor_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl + latent_action_kl
                        anchor_kl = discrete_anchor_kl + latent_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "global":
                        latent_global_kl = latent_kl(
                            mu1=goal_emb_cond_x_mu,
                            logvar1=goal_emb_cond_x_logvar,
                            mu2=goal_emb_mu,
                            logvar2=goal_emb_logvar,
                            reduction="batchmean",
                        )

                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl

                    elif self.hybrid_cond_pzX_regularize_type == "none":
                        action_kl = discrete_action_kl
                        anchor_kl = discrete_anchor_kl
                    else:
                        raise ValueError(
                            f"hybrid_cond_pzX_regularize_type={self.hybrid_cond_pzX_regularize_type} not supported"
                        )

                else:
                    raise ValueError(
                        f"goal_emb_cond_x_loss_type={self.goal_emb_cond_x_loss_type} not supported for {self.model_with_cond_x.conditioning}"
                    )

                if discrete_action_kl is not None:
                    log_values[loss_prefix + "discrete_action_kl"] = discrete_action_kl
                if discrete_anchor_kl is not None:
                    log_values[loss_prefix + "discrete_anchor_kl"] = discrete_anchor_kl
                if latent_action_kl is not None:
                    log_values[loss_prefix + "latent_action_kl"] = latent_action_kl
                if latent_anchor_kl is not None:
                    log_values[loss_prefix + "latent_anchor_kl"] = latent_anchor_kl
                if latent_global_kl is not None:
                    log_values[loss_prefix + "latent_global_kl"] = latent_global_kl

            else:
                raise ValueError(
                    f"conditioning={self.model_with_cond_x.conditioning} not supported"
                )

            if self.model_with_cond_x.conditioning in [
                "latent_z_linear",
                "latent_z_linear_internalcond",
            ]:
                goal_emb_loss = latent_goal_kl
            else:
                goal_emb_loss = action_kl + anchor_kl
                if self.hybrid_cond_pzX_regularize_type == "global":
                    goal_emb_loss += latent_global_kl

            if self.overwrite_loss:
                # Only update p(z|X) encoder for p(z|X) pass
                loss = self.goal_emb_cond_x_loss_weight * goal_emb_loss
            else:
                # Update p(z|X) encoder and TAXPose decoder for p(z|X) pass
                loss += self.goal_emb_cond_x_loss_weight * goal_emb_loss

            log_values[loss_prefix + "goal_emb_cond_x_loss"] = (
                self.goal_emb_cond_x_loss_weight * goal_emb_loss
            )
            log_values[loss_prefix + "action_kl"] = action_kl
            log_values[loss_prefix + "anchor_kl"] = anchor_kl

        return loss, log_values

    def extract_flow_and_weight(self, *args, **kwargs):
        return self.training_module_no_cond_x.extract_flow_and_weight(*args, **kwargs)

    def module_step(self, batch, batch_idx, log_prefix=""):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        points_onetrans_action = (
            batch["points_action_onetrans"]
            if "points_action_onetrans" in batch
            else batch["points_action"]
        )
        points_onetrans_anchor = (
            batch["points_anchor_onetrans"]
            if "points_anchor_onetrans" in batch
            else batch["points_anchor"]
        )

        # If joint training prior
        if self.joint_train_prior:
            # Unfreeze components for p(z|Y) pass
            self.training_module_no_cond_x.model.freeze_residual_flow = False
            self.training_module_no_cond_x.model.freeze_z_embnn = False
            self.training_module_no_cond_x.model.freeze_embnn = False
            self.training_module_no_cond_x.model.tax_pose.freeze_embnn = False

            # p(z|Y) pass
            pzY_loss, pzY_log_values = self.training_module_no_cond_x.module_step(
                batch, batch_idx
            )

            # Potentially freeze components for p(z|X) pass
            self.training_module_no_cond_x.model.freeze_residual_flow = (
                self.cfg_freeze_residual_flow
            )
            self.training_module_no_cond_x.model.freeze_z_embnn = (
                self.cfg_freeze_z_embnn
            )
            self.training_module_no_cond_x.model.freeze_embnn = self.cfg_freeze_embnn
            self.training_module_no_cond_x.model.tax_pose.freeze_embnn = (
                self.cfg_freeze_embnn
            )

        # Debugging, optionally use the exact same z samples for p(z|X) as selected by p(z|Y)
        z_samples = None
        if self.pzX_use_pzY_z_samples:
            model_no_cond_x_outputs = self.model(
                points_trans_action,
                points_trans_anchor,
                points_onetrans_action,
                points_onetrans_anchor,
                n_samples=self.n_samples,
            )
            translation_samples_action = [
                model_no_cond_x_outputs[i]["trans_sample_action"]
                for i in range(len(model_no_cond_x_outputs))
            ]
            translation_samples_anchor = [
                model_no_cond_x_outputs[i]["trans_sample_anchor"]
                for i in range(len(model_no_cond_x_outputs))
            ]
            z_samples = {
                "translation_samples_action": translation_samples_action,
                "translation_samples_anchor": translation_samples_anchor,
            }

        # Do the p(z|X) pass, determine whether continuous latent z is sampled
        sample_latent = (
            self.model_with_cond_x.conditioning
            in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]
            and self.hybrid_cond_pzX_sample_latent
        )
        model_outputs = self.model_with_cond_x(
            points_trans_action,
            points_trans_anchor,
            points_onetrans_action,
            points_onetrans_anchor,
            sampling_method="gumbel",
            n_samples=self.n_samples,
            z_samples=z_samples,
            sample_latent=sample_latent,
        )

        # Compute p(z|X) losses
        log_values = {}
        log_prefix = "pzX_" if self.joint_train_prior else log_prefix
        loss, log_values = self.compute_loss(
            model_outputs, batch, log_values=log_values, loss_prefix=log_prefix
        )

        # If joint training prior also use p(z|Y) losses
        if self.joint_train_prior:
            loss = pzY_loss + loss
            log_values = {**pzY_log_values, **log_values}

        # Auxiliary InfoNCE loss
        # Only do this once regardless of n_samples
        if self.joint_infonce_loss_weight > 0:
            # One encoder
            if self.model_with_cond_x.encoder_type in ["1_dgcnn"]:
                raise NotImplementedError(
                    "joint_pretraining_loss not implemented for 1_dgcnn"
                )
            # Two encoders
            else:
                model_with_cond_x_input_dims = self.model_with_cond_x.input_dims
                action_infonce_loss, action_infonce_log_values = compute_infonce_loss(
                    model=self.model_with_cond_x.p_z_cond_x_embnn_action,
                    points=points_trans_action[:, :, :model_with_cond_x_input_dims],
                    log_prefix=log_prefix + "action_",
                )
                anchor_infonce_loss, anchor_infonce_log_values = compute_infonce_loss(
                    model=self.model_with_cond_x.p_z_cond_x_embnn_anchor,
                    points=points_trans_anchor[:, :, :model_with_cond_x_input_dims],
                    log_prefix=log_prefix + "anchor_",
                )

                loss += self.joint_infonce_loss_weight * (
                    action_infonce_loss + anchor_infonce_loss
                )
                log_values = {
                    **log_values,
                    **action_infonce_log_values,
                    **anchor_infonce_log_values,
                }

        # Debugging, get metrics when using more than 1 sample
        if self.get_errors_across_samples:
            with torch.no_grad():
                T0 = Transform3d(matrix=batch["T0"])
                T1 = Transform3d(matrix=batch["T1"])
                T_aug_list = (
                    [Transform3d(matrix=T_aug) for T_aug in batch["T_aug_list"]]
                    if "T_aug_list" in batch
                    else None
                )

                if self.use_debug_sampling_methods:
                    sampling_settings = [("random", 1), ("top_n", 3)]
                else:
                    sampling_settings = [("gumbel", self.n_samples)]

                all_model_outputs = []
                for sampling_setting in sampling_settings:
                    error_model_outputs = self.model_with_cond_x(
                        points_trans_action,
                        points_trans_anchor,
                        points_onetrans_action,
                        points_onetrans_anchor,
                        sampling_method=sampling_setting[0],
                        n_samples=sampling_setting[1],
                    )
                    all_model_outputs.append(error_model_outputs)

                all_predictions = []
                for i in range(len(all_model_outputs)):
                    setting_predictions = []
                    for j in range(len(all_model_outputs[i])):
                        x_action = all_model_outputs[i][j]["flow_action"]
                        x_anchor = all_model_outputs[i][j]["flow_anchor"]
                        goal_emb = all_model_outputs[i][j]["goal_emb"]
                        goal_emb_cond_x = all_model_outputs[i][j]["goal_emb_cond_x"]

                        # If we've applied some sampling, we need to extract the predictions too...
                        if "sampled_ixs_action" in error_model_outputs[i][j]:
                            ixs_action = error_model_outputs[i][j][
                                "sampled_ixs_action"
                            ].unsqueeze(-1)
                            sampled_points_trans_action = torch.take_along_dim(
                                points_trans_action, ixs_action, dim=1
                            )
                        else:
                            sampled_points_trans_action = points_trans_action

                        if "sampled_ixs_anchor" in error_model_outputs[i][j]:
                            ixs_anchor = error_model_outputs[i][j][
                                "sampled_ixs_anchor"
                            ].unsqueeze(-1)
                            sampled_points_trans_anchor = torch.take_along_dim(
                                points_trans_anchor, ixs_anchor, dim=1
                            )
                        else:
                            sampled_points_trans_anchor = points_trans_anchor

                        # Get flow/weights
                        pred_flow_action, pred_w_action = self.extract_flow_and_weight(
                            x_action
                        )
                        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
                            x_anchor
                        )

                        # Get predicted transform
                        pred_T_action = dualflow2pose(
                            xyz_src=sampled_points_trans_action,
                            xyz_tgt=sampled_points_trans_anchor,
                            flow_src=pred_flow_action,
                            flow_tgt=pred_flow_anchor,
                            weights_src=pred_w_action,
                            weights_tgt=pred_w_anchor,
                            return_transform3d=True,
                            normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                            temperature=self.training_module_no_cond_x.softmax_temperature,
                        )

                        setting_predictions.append(pred_T_action)
                    all_predictions.append(setting_predictions)

                for i in range(len(all_predictions)):
                    (
                        error_R_maxs,
                        error_R_mins,
                        error_R_means,
                        error_t_maxs,
                        error_t_mins,
                        error_t_means,
                    ) = get_all_sample_errors(
                        all_predictions[i],
                        T0,
                        T1,
                        mode=self.training_module_no_cond_x.error_mode_2rack,
                        T_aug_list=T_aug_list,
                    )
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_mean"
                    ] = torch.mean(torch.Tensor(error_R_means))
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_mean"
                    ] = torch.mean(torch.Tensor(error_t_means))
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_max"
                    ] = torch.max(torch.Tensor(error_R_maxs))
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_max"
                    ] = torch.max(torch.Tensor(error_t_maxs))
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_R_min"
                    ] = torch.min(torch.Tensor(error_R_mins))
                    log_values[
                        log_prefix
                        + f"{sampling_settings[i][0]}-{sampling_settings[i][1]}_error_t_min"
                    ] = torch.min(torch.Tensor(error_t_mins))

        # Return outputs when in adversarial mode
        if self.pzX_adversarial:
            return loss, log_values, model_outputs
        return loss, log_values

    def visualize_results(self, batch, batch_idx, log_prefix=""):
        res_images = self.training_module_no_cond_x.visualize_results(
            batch, batch_idx, log_prefix="pzY_"
        )

        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        points_onetrans_action = (
            batch["points_action_onetrans"]
            if "points_action_onetrans" in batch
            else batch["points_action"]
        )
        points_onetrans_anchor = (
            batch["points_anchor_onetrans"]
            if "points_anchor_onetrans" in batch
            else batch["points_anchor"]
        )

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        # Debugging, optionally use the exact same z samples for p(z|X) as selected by p(z|Y)
        z_samples = None
        if self.pzX_use_pzY_z_samples:
            model_no_cond_x_outputs = self.model(
                points_trans_action,
                points_trans_anchor,
                points_onetrans_action,
                points_onetrans_anchor,
                n_samples=self.n_samples,
            )
            translation_samples_action = [
                model_no_cond_x_outputs[i]["trans_sample_action"]
                for i in range(len(model_no_cond_x_outputs))
            ]
            translation_samples_anchor = [
                model_no_cond_x_outputs[i]["trans_sample_anchor"]
                for i in range(len(model_no_cond_x_outputs))
            ]
            z_samples = {
                "translation_samples_action": translation_samples_action,
                "translation_samples_anchor": translation_samples_anchor,
            }

        # Do the p(z|X) pass, determine whether continuous latent z is sampled
        sample_latent = (
            self.model_with_cond_x.conditioning
            in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
                "hybrid_pos_delta_l2norm_global",
                "hybrid_pos_delta_l2norm_global_internalcond",
            ]
            and self.hybrid_cond_pzX_sample_latent
        )
        model_outputs = self.model_with_cond_x(
            points_trans_action,
            points_trans_anchor,
            points_onetrans_action,
            points_onetrans_anchor,
            sampling_method="gumbel",
            n_samples=1,
            z_samples=z_samples,
            sample_latent=sample_latent,
        )
        goal_emb = model_outputs[0]["goal_emb"]
        goal_emb_cond_x = model_outputs[0]["goal_emb_cond_x"]

        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]
        points_onetrans_action = points_onetrans_action[:, :, :3]
        points_onetrans_anchor = points_onetrans_anchor[:, :, :3]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_outputs[0]:
            ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_action = torch.take_along_dim(
                points_action, ixs_action, dim=1
            )
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_action = points_action
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_outputs[0]:
            ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_anchor = torch.take_along_dim(
                points_anchor, ixs_anchor, dim=1
            )
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_anchor = points_anchor
            sampled_points_trans_anchor = points_trans_anchor

        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_outputs[0]["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            x_anchor = model_outputs[0]["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action,
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_outputs[0]["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

            # Get predicted transform
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_outputs[0]["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature,
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(
                f"ERROR: Invalid flow supervision type: {self.flow_supervision}"
            )

        if self.decoder_type == "taxpose":
            pred_points_action = pred_T_action.transform_points(points_trans_action)
        elif self.decoder_type in ["flow", "point"] and self.flow_frame == "aligned":
            action_center = model_outputs[0]["trans_pt_action"][:, :, None]
            anchor_center = model_outputs[0]["trans_pt_anchor"][:, :, None]

            points_trans_action_in_aligned_frame = (
                points_trans_action - action_center.permute(0, 2, 1)
            )
            points_action_pred_in_aligned_frame = pred_T_action.transform_points(
                points_trans_action_in_aligned_frame
            )
            pred_points_action = (
                points_action_pred_in_aligned_frame + anchor_center.permute(0, 2, 1)
            )
        else:
            raise ValueError(f"ERROR: Invalid decoder type: {self.decoder_type}")

        demo_points_tensors = [points_onetrans_action[0], points_onetrans_anchor[0]]
        demo_points_colors = ["blue", "red"]
        if "points_action_aug_trans" in batch:
            demo_points_tensors.append(batch["points_action_aug_trans"][0, :, :3])
            demo_points_colors.append("yellow")
        demo_points = get_color(
            tensor_list=demo_points_tensors, color_list=demo_points_colors
        )
        res_images[log_prefix + "demo_points"] = wandb.Object3D(demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "action_transformed_action"] = wandb.Object3D(
            action_transformed_action
        )

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "anchor_transformed_anchor"] = wandb.Object3D(
            anchor_transformed_anchor
        )

        # transformed_input_points = get_color(tensor_list=[
        #     points_trans_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        # res_images[log_prefix+'transformed_input_points'] = wandb.Object3D(
        #     transformed_input_points)

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "demo_points_apply_action_transform"] = wandb.Object3D(
            demo_points_apply_action_transform
        )

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=["blue", "red"],
        )
        res_images[log_prefix + "apply_action_transform_demo_comparable"] = (
            wandb.Object3D(apply_action_transform_demo_comparable)
        )

        predicted_vs_gt_transform_tensors = [
            T1.inverse().transform_points(pred_points_action)[0],
            points_action[0],
            T1.inverse().transform_points(points_trans_anchor)[0],
        ]
        predicted_vs_gt_transform_colors = ["blue", "green", "red"]
        if "points_action_aug_trans" in batch:
            predicted_vs_gt_transform_tensors.append(
                batch["points_action_aug_trans"][0, :, :3]
            )
            predicted_vs_gt_transform_colors.append("yellow")
        predicted_vs_gt_transform_applied = get_color(
            tensor_list=predicted_vs_gt_transform_tensors,
            color_list=predicted_vs_gt_transform_colors,
        )
        res_images[log_prefix + "predicted_vs_gt_transform_applied"] = wandb.Object3D(
            predicted_vs_gt_transform_applied
        )

        apply_predicted_transform = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=[
                "blue",
                "orange",
                "red",
            ],
        )
        res_images[log_prefix + "apply_predicted_transform"] = wandb.Object3D(
            apply_predicted_transform
        )

        # loss_points_action = get_color(
        #     tensor_list=[points_action_target[0], pred_points_action[0]], color_list=['green', 'red'])
        # res_images[log_prefix+'loss_points_action'] = wandb.Object3D(
        #     loss_points_action)

        pred_w_points_list = []
        pred_w_colors_list = []
        if self.flow_supervision in ["both", "action2anchor"]:
            pred_w_points_list.append(sampled_points_action[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_action[0]))
        if self.flow_supervision in ["both", "anchor2action"]:
            pred_w_points_list.append(sampled_points_anchor[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_anchor[0]))

        pred_w_on_objects = np.concatenate(
            [
                torch.cat(pred_w_points_list, dim=0).cpu().numpy(),
                np.concatenate(pred_w_colors_list, axis=0),
            ],
            axis=-1,
        )

        res_images[log_prefix + "pred_w"] = wandb.Object3D(
            pred_w_on_objects, markerSize=1000
        )

        # This visualization only applies to methods that have discrete per-point latents
        if self.model.conditioning not in [
            "latent_z_linear",
            "latent_z_linear_internalcond",
        ]:
            if self.model.conditioning in [
                "hybrid_pos_delta_l2norm",
                "hybrid_pos_delta_l2norm_internalcond",
            ]:
                goal_emb_cond_x = goal_emb_cond_x[:, :1, :]

            # Plot goal embeddings on objects
            goal_emb_cond_x_norm_action = (
                F.softmax(goal_emb_cond_x[0, :, : points_action.shape[1]], dim=-1)
                .detach()
                .cpu()
            )
            goal_emb_cond_x_norm_anchor = (
                F.softmax(goal_emb_cond_x[0, :, points_action.shape[1] :], dim=-1)
                .detach()
                .cpu()
            )

            colors_action = color_gradient(goal_emb_cond_x_norm_action[0])
            colors_anchor = color_gradient(goal_emb_cond_x_norm_anchor[0])
            points = (
                torch.cat([points_action[0].detach(), points_anchor[0].detach()], dim=0)
                .cpu()
                .numpy()
            )
            goal_emb_on_objects = np.concatenate(
                [points, np.concatenate([colors_action, colors_anchor], axis=0)],
                axis=-1,
            )

            res_images["goal_emb_cond_x"] = wandb.Object3D(
                goal_emb_on_objects, markerSize=1000
            )  # marker_scale * range_size)

            # Plot the p(z|Y) and p(z|X) goal embeddings as bar plots
            if self.plot_encoder_distribution:
                # Get embeddings for p(z|Y2)
                if "points_action_aug_trans" in batch:
                    points_action_aug_trans = batch["points_action_aug_trans"]
                    points_trans_action = batch["points_action_trans"]
                    points_trans_anchor = batch["points_anchor_trans"]
                    points_action = batch["points_action"]
                    points_anchor = batch["points_anchor"]
                    points_onetrans_action = (
                        batch["points_action_onetrans"]
                        if "points_action_onetrans" in batch
                        else batch["points_action"]
                    )
                    points_onetrans_anchor = (
                        batch["points_anchor_onetrans"]
                        if "points_anchor_onetrans" in batch
                        else batch["points_anchor"]
                    )

                    model_no_cond_x_outputs = self.model(
                        points_trans_action,
                        points_trans_anchor,
                        points_action_aug_trans,
                        points_onetrans_anchor,
                        n_samples=self.n_samples,
                    )

                    pzY2_emb = model_no_cond_x_outputs[0]["goal_emb"]
                    if self.model.conditioning in [
                        "hybrid_pos_delta_l2norm",
                        "hybrid_pos_delta_l2norm_internalcond",
                    ]:
                        pzY2_emb = pzY2_emb[:, :1, :]

                    pzY2_action_emb = pzY2_emb[0, :, : points_action.shape[1]]
                    pzY2_anchor_emb = pzY2_emb[0, :, points_action.shape[1] :]
                    pzY2_action_dist = (
                        F.softmax(pzY2_action_emb, dim=-1).detach().cpu().numpy()[0]
                    )
                    pzY2_anchor_dist = (
                        F.softmax(pzY2_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                    )

                    pzY2_actions_probs = np.array([prob for prob in pzY2_action_dist])
                    pzY2_anchor_probs = np.array([prob for prob in pzY2_anchor_dist])

                pzY1_action_emb = goal_emb[0, :, : points_action.shape[1]]
                pzY1_anchor_emb = goal_emb[0, :, points_action.shape[1] :]
                pzY1_action_dist = (
                    F.softmax(pzY1_action_emb, dim=-1).detach().cpu().numpy()[0]
                )
                pzY1_anchor_dist = (
                    F.softmax(pzY1_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                )

                pzX_action_dist = goal_emb_cond_x_norm_action.numpy()[0]
                pzX_anchor_dist = goal_emb_cond_x_norm_anchor.numpy()[0]

                x_vals = np.arange(pzY1_action_dist.shape[0])
                pzY1_actions_probs = np.array([prob for prob in pzY1_action_dist])
                pzY1_anchor_probs = np.array([prob for prob in pzY1_anchor_dist])

                pzX_actions_probs = np.array([prob for prob in pzX_action_dist])
                pzX_anchor_probs = np.array([prob for prob in pzX_anchor_dist])

                layout_margin = go.layout.Margin(
                    l=50,  # left margin
                    r=120,  # right margin
                    b=50,  # bottom margin
                    t=50,  # top margin
                    autoexpand=False,
                )

                # Plot action distributions
                action_max_prob = max(
                    np.max(pzY1_actions_probs), np.max(pzX_actions_probs)
                )

                pzY_action_data = [
                    go.Bar(
                        name="pzY1",
                        x=x_vals,
                        y=pzY1_actions_probs,
                        width=1,
                        marker_color="blue",
                        opacity=0.5,
                        showlegend=True,
                    )
                ]
                if "points_action_aug_trans" in batch:
                    pzY_action_data.append(
                        go.Bar(
                            name="pzY2",
                            x=x_vals,
                            y=pzY2_actions_probs,
                            width=1,
                            marker_color="red",
                            opacity=0.5,
                            showlegend=True,
                        )
                    )
                    action_max_prob = max(action_max_prob, np.max(pzY2_actions_probs))

                pzY_action_plot = go.Figure(data=pzY_action_data)
                pzY_action_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, action_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                pzX_action_plot = go.Figure(
                    data=[
                        go.Bar(
                            name="pzX",
                            x=x_vals,
                            y=pzX_actions_probs,
                            width=1,
                            marker_color="green",
                            opacity=1,
                            showlegend=True,
                        ),
                    ]
                )
                pzX_action_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, action_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                # Plot anchor distributions
                anchor_max_prob = max(
                    np.max(pzY1_anchor_probs), np.max(pzX_anchor_probs)
                )

                pzY_anchor_data = [
                    go.Bar(
                        name="pzY1",
                        x=x_vals,
                        y=pzY1_anchor_probs,
                        width=1,
                        marker_color="blue",
                        opacity=0.5,
                        showlegend=True,
                    )
                ]
                if "points_action_aug_trans" in batch:
                    pzY_anchor_data.append(
                        go.Bar(
                            name="pzY2",
                            x=x_vals,
                            y=pzY2_anchor_probs,
                            width=1,
                            marker_color="red",
                            opacity=0.5,
                            showlegend=True,
                        )
                    )
                    anchor_max_prob = max(anchor_max_prob, np.max(pzY2_anchor_probs))

                pzY_anchor_plot = go.Figure(data=pzY_anchor_data)
                pzY_anchor_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, anchor_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                pzX_anchor_plot = go.Figure(
                    data=[
                        go.Bar(
                            name="pzX",
                            x=x_vals,
                            y=pzX_anchor_probs,
                            width=1,
                            marker_color="green",
                            opacity=1,
                            showlegend=True,
                        ),
                    ]
                )
                pzX_anchor_plot.update_layout(
                    barmode="overlay",
                    height=480,
                    width=1920,
                    yaxis_range=[0, anchor_max_prob * 1.1],
                    margin=layout_margin,
                    legend={"entrywidth": 40},
                )

                res_images[log_prefix + "pzY_action_distribution"] = wandb.Html(
                    plotly.io.to_html(pzY_action_plot, include_plotlyjs="cdn")
                )
                res_images[log_prefix + "pzX_action_distribution"] = wandb.Html(
                    plotly.io.to_html(pzX_action_plot, include_plotlyjs="cdn")
                )
                res_images[log_prefix + "pzY_anchor_distribution"] = wandb.Html(
                    plotly.io.to_html(pzY_anchor_plot, include_plotlyjs="cdn")
                )
                res_images[log_prefix + "pzX_anchor_distribution"] = wandb.Html(
                    plotly.io.to_html(pzX_anchor_plot, include_plotlyjs="cdn")
                )

        return res_images
