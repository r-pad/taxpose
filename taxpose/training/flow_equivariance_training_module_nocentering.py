import torch
import wandb
from pytorch3d.transforms import Transform3d
from torch import nn
from torchvision.transforms import ToTensor

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.color_utils import get_color
from taxpose.utils.se3 import (
    dense_flow_loss,
    flow2pose,
    dualflow2pose,
    get_degree_angle,
    get_translation,
)

mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()


class EquivarianceTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model=None,
        lr=1e-3,
        image_log_period=500,
        action_weight=1,
        anchor_weight=1,
        displace_loss_weight=1,
        consistency_loss_weight=0.1,
        direct_correspondence_loss_weight=1,
        return_flow_component=False,
        weight_normalize="l1",
        sigmoid_on=False,
        softmax_temperature=None,
        flow_supervision='both'  # ('both', 'action2anchor', 'anchor2action')
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
        )
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.displace_loss_weight = displace_loss_weight
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.direct_correspondence_loss_weight = direct_correspondence_loss_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize

        self.return_flow_component = return_flow_component
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        self.flow_supervision = flow_supervision
        if self.weight_normalize == "l1":
            assert self.sigmoid_on, "l1 weight normalization need sigmoid on"

    def compute_loss(self, x_action, x_anchor, batch, log_values={}, loss_prefix=""):
        points_action = batch["points_action"][:, :, :3]  # action point clouds
        points_anchor = batch["points_anchor"][:, :, :3]  # anchor point clouds
        # action point clouds transformed by T0
        points_trans_action = batch["points_action_trans"][:, :, :3]
        # anchor point clouds transformed by T1
        points_trans_anchor = batch["points_anchor_trans"][:, :, :3]

        # SE(3) transformation applied to points_action
        T0 = Transform3d(matrix=batch["T0"])
        # SE(3) transformation applied to points_anchor
        T1 = Transform3d(matrix=batch["T1"])

        R0_max, R0_min, R0_mean = get_degree_angle(
            T0)  # rotation component applied to points_action
        R1_max, R1_min, R1_mean = get_degree_angle(
            T1)  # rotation component applied to points_anchor
        t0_max, t0_min, t0_mean = get_translation(
            T0)  # translation component applied to points_action
        t1_max, t1_min, t1_mean = get_translation(
            T1)  # translation component applied to points_anchor

        pred_flow_action, pred_w_action = self.extract_flow_and_weight(
            x_action)  # flow predicted from action to anchor, per point importance weight for action points
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
            x_anchor)  # flow predicted from anchor to action, per point importance weight for anchor points

        if self.flow_supervision == 'both':
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

            induced_flow_action = (
                pred_T_action.transform_points(
                    points_trans_action) - points_trans_action
            ).detach()
            pred_points_action = pred_T_action.transform_points(
                points_trans_action)

            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            points_action_target = T1.transform_points(points_action)

            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

            error_t_max, error_t_min, error_t_mean = get_translation(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

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
                points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
            )

            pred_T_anchor = pred_T_action.inverse()

            induced_flow_anchor = (pred_T_anchor.transform_points(
                points_trans_anchor) - points_trans_anchor).detach()
            pred_points_anchor = pred_T_anchor.transform_points(
                points_trans_anchor)

            # pred_T_action=T1T0^-1
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
            dense_loss_anchor = dense_flow_loss(points=points_trans_anchor,
                                                flow_pred=pred_flow_anchor,
                                                trans_gt=gt_T_anchor)
            self.action_weight = (self.action_weight) / \
                (self.action_weight+self.anchor_weight)
            self.anchor_weight = (self.anchor_weight) / \
                (self.action_weight+self.anchor_weight)
        elif self.flow_supervision == 'action2anchor':

            pred_T_action = flow2pose(xyz=points_trans_action,
                                      flow=pred_flow_action,
                                      weights=pred_w_action,
                                      return_transform3d=True,
                                      normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)
            induced_flow_action = (pred_T_action.transform_points(
                points_trans_action) - points_trans_action).detach()
            pred_points_action = pred_T_action.transform_points(
                points_trans_action)

            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            points_action_target = T1.transform_points(points_action)

            error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
                T1).compose(pred_T_action.inverse()))

            error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
                T1).compose(pred_T_action.inverse()))

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

            dense_loss_action = dense_flow_loss(points=points_trans_action,
                                                flow_pred=pred_flow_action,
                                                trans_gt=gt_T_action)

            # Zero anchor terms
            self.anchor_weight = 0
            self.action_weight = (self.action_weight) / \
                (self.action_weight+self.anchor_weight)
            point_loss_anchor = 0
            smoothness_loss_anchor = 0
            dense_loss_anchor = 0
        elif self.flow_supervision == 'anchor2action':
            pred_T_anchor = flow2pose(xyz=points_trans_anchor,
                                      flow=pred_flow_anchor,
                                      weights=pred_w_anchor,
                                      return_transform3d=True,
                                      normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)
            induced_flow_anchor = (pred_T_anchor.transform_points(
                points_trans_anchor) - points_trans_anchor).detach()
            pred_points_anchor = pred_T_anchor.transform_points(
                points_trans_anchor)

            # pred_T_action=T1T0^-1
            gt_T_anchor = T1.inverse().compose(T0)
            points_anchor_target = T0.transform_points(points_anchor)

            error_R_max, error_R_min, error_R_mean = get_degree_angle(T1.inverse().compose(
                T0).compose(pred_T_anchor.inverse()))

            error_t_max, error_t_min, error_t_mean = get_translation(T1.inverse().compose(
                T0).compose(pred_T_anchor.inverse()))

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
            dense_loss_anchor = dense_flow_loss(points=points_trans_anchor,
                                                flow_pred=pred_flow_anchor,
                                                trans_gt=gt_T_anchor)
            # Zero action terms
            self.action_weight = 0
            self.anchor_weight = (self.anchor_weight) / \
                (self.action_weight+self.anchor_weight)
            point_loss_action = 0
            smoothness_loss_action = 0
            dense_loss_action = 0

        point_loss = self.action_weight * point_loss_action + \
            self.anchor_weight * point_loss_anchor

        dense_loss = self.action_weight*dense_loss_action + \
            self.anchor_weight*dense_loss_anchor

        smoothness_loss = self.action_weight * smoothness_loss_action + \
            self.anchor_weight * smoothness_loss_anchor

        loss = (
            self.displace_loss_weight * point_loss
            + self.consistency_loss_weight * smoothness_loss
            + self.direct_correspondence_loss_weight * dense_loss
        )

        log_values[loss_prefix +
                   "point_loss"] = self.displace_loss_weight * point_loss
        log_values[loss_prefix + "smoothness_loss"] = (
            self.consistency_loss_weight * smoothness_loss
        )
        log_values[loss_prefix +
                   "dense_loss"] = self.direct_correspondence_loss_weight * dense_loss

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

        log_values[loss_prefix + "error_R_mean"] = error_R_mean
        log_values[loss_prefix + "error_t_mean"] = error_t_mean

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

    def module_step(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])
        if self.return_flow_component:
            model_output = self.model(points_trans_action, points_trans_anchor)
            x_action = model_output["flow_action"]
            x_anchor = model_output["flow_anchor"]
            residual_flow_action = model_output["residual_flow_action"]
            residual_flow_anchor = model_output["residual_flow_anchor"]
            corr_flow_action = model_output["corr_flow_action"]
            corr_flow_anchor = model_output["corr_flow_anchor"]
        else:
            x_action, x_anchor = self.model(
                points_trans_action, points_trans_anchor)

        log_values = {}
        loss, log_values = self.compute_loss(
            x_action, x_anchor, batch, log_values=log_values, loss_prefix=""
        )
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        # classes = batch['classes']
        # points = batch['points']
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        # points_trans = batch['points_trans']
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        if self.return_flow_component:
            model_output = self.model(points_trans_action, points_trans_anchor)
            x_action = model_output["flow_action"]
            x_anchor = model_output["flow_anchor"]
            residual_flow_action = model_output["residual_flow_action"]
            residual_flow_anchor = model_output["residual_flow_anchor"]
            corr_flow_action = model_output["corr_flow_action"]
            corr_flow_anchor = model_output["corr_flow_anchor"]
        else:
            x_action, x_anchor = self.model(
                points_trans_action, points_trans_anchor)

        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]

        pred_flow_action = x_action[:, :, :3]
        if x_action.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_action = torch.sigmoid(x_action[:, :, 3])
            else:
                pred_w_action = x_action[:, :, 3]
        else:
            pred_w_action = None

        pred_flow_anchor = x_anchor[:, :, :3]
        if x_anchor.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_anchor = torch.sigmoid(x_anchor[:, :, 3])
            else:
                pred_w_anchor = x_anchor[:, :, 3]
        else:
            pred_w_anchor = None
        if self.flow_supervision == 'both':
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
        elif self.flow_supervision == 'action2anchor':
            pred_T_action = flow2pose(xyz=points_trans_action,
                                      flow=pred_flow_action,
                                      weights=pred_w_action,
                                      return_transform3d=True,
                                      normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)
            pred_T_anchor = pred_T_action.inverse()

        elif self.flow_supervision == 'anchor2action':
            pred_T_anchor = flow2pose(xyz=points_trans_anchor,
                                      flow=pred_flow_anchor,
                                      weights=pred_w_anchor,
                                      return_transform3d=True,
                                      normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)
            pred_T_action = pred_T_anchor.inverse()

        pred_points_action = pred_T_action.transform_points(
            points_trans_action)
        points_action_target = T1.transform_points(points_action)

        res_images = {}

        demo_points = get_color(
            tensor_list=[points_action[0], points_anchor[0]
                         ], color_list=["blue", "red"]
        )
        res_images["demo_points"] = wandb.Object3D(demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]],
            color_list=["blue", "red"],
        )
        res_images["action_transformed_action"] = wandb.Object3D(
            action_transformed_action
        )

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["anchor_transformed_anchor"] = wandb.Object3D(
            anchor_transformed_anchor
        )

        transformed_input_points = get_color(
            tensor_list=[points_trans_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["transformed_input_points"] = wandb.Object3D(
            transformed_input_points
        )

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["demo_points_apply_action_transform"] = wandb.Object3D(
            demo_points_apply_action_transform
        )

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=["blue", "red"],
        )
        res_images["apply_action_transform_demo_comparable"] = wandb.Object3D(
            apply_action_transform_demo_comparable
        )

        predicted_vs_gt_transform_applied = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                points_action[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=[
                "blue",
                "green",
                "red",
            ],
        )
        res_images["predicted_vs_gt_transform_applied"] = wandb.Object3D(
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
        res_images["apply_predicted_transform"] = wandb.Object3D(
            apply_predicted_transform
        )

        loss_points_action = get_color(
            tensor_list=[points_action_target[0], pred_points_action[0]],
            color_list=["green", "red"],
        )
        res_images["loss_points_action"] = wandb.Object3D(loss_points_action)

        return res_images
