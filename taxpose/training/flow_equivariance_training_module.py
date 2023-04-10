import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import Transform3d, matrix_to_axis_angle  # se3_log_map
from torch import nn
from torchvision.transforms import ToTensor

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.display_headless import quiver3d, scatter3d  # type: ignore
from taxpose.utils.se3 import flow2pose, get_degree_angle, get_translation

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
        smoothness_weight=0.1,
        chamfer_weight=10000,
        point_loss_type=0,
        return_flow_component=False,
        weight_normalize="l1",
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
        self.chamfer_weight = chamfer_weight
        self.display_action = True
        self.display_anchor = True
        # 0 for mse loss, 1 for chamfer distance, 2 for mse loss + chamfer distance
        self.point_loss_type = point_loss_type
        self.return_flow_component = return_flow_component
        self.weight_normalize = weight_normalize

    def get_degree_angle(self, T):
        axis_angle_T = matrix_to_axis_angle(T.get_matrix()[:, :3, :3])  # B,3
        angle_rad_T = torch.norm(axis_angle_T, dim=1) * 180 / np.pi  # B
        max = torch.max(angle_rad_T).item()
        min = torch.min(angle_rad_T).item()
        mean = torch.mean(angle_rad_T).item()
        return max, min, mean

    def get_translation(self, T):
        t = T.get_matrix()[:, 3:, :3]  # B,3
        t_norm = torch.norm(t, dim=1)  # B
        max = torch.max(t_norm).item()
        min = torch.min(t_norm).item()
        mean = torch.mean(t_norm).item()
        return max, min, mean

    def cal_loss(self, x_action, x_anchor, batch, log_values={}, loss_prefix=""):
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        R0_max, R0_min, R0_mean = get_degree_angle(T0)
        R1_max, R1_min, R1_mean = get_degree_angle(T1)
        t0_max, t0_min, t0_mean = get_translation(T0)
        t1_max, t1_min, t1_mean = get_translation(T1)

        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        # Transfrom predicted using both flows.
        # Currently not useing this. Could replace pred_T_action with this.
        # pred_flow = torch.cat([pred_flow_action, pred_flow_anchor])
        # polarity = torch.cat([torch.ones_like(pred_w_action), torch.zeros_like(pred_w_anchor)])
        # pred_w = torch.cat([pred_w_action, pred_w_anchor])
        # pred_T = dualflow2pose(points_trans, pred_flow, polarity,
        #     weights=pred_w, return_transform3d=True)

        # Transfrom predicted from X_0 to X_1
        pred_T_action = flow2pose(
            points_trans_action,
            pred_flow_action,
            pred_w_action,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        induced_flow_action = (
            pred_T_action.transform_points(points_trans_action) - points_trans_action
        ).detach()
        pred_points_action = pred_T_action.compose(T1.inverse()).transform_points(
            points_trans_action
        )

        # Transfrom predicted from X_1 to X_0
        # Should be pred_T_0^-1
        # Could rewrite the action points s.t. we don't need the inverse
        pred_T_anchor = flow2pose(
            points_trans_anchor,
            pred_flow_anchor,
            pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        induced_flow_anchor = (
            pred_T_anchor.transform_points(points_trans_anchor) - points_trans_anchor
        ).detach()
        pred_points_anchor = pred_T_anchor.compose(T0.inverse()).transform_points(
            points_trans_anchor
        )

        if self.point_loss_type == 0:
            # mse loss
            point_loss_action = mse_criterion(
                pred_points_action,
                points_action,
            )
            point_loss_anchor = mse_criterion(
                pred_points_anchor,
                points_anchor,
            )
        elif self.point_loss_type == 1:
            # chamfer loss
            point_loss_action = (
                self.chamfer_weight
                * chamfer_distance(
                    pred_points_action,
                    points_action,
                )[0].item()
            )
            point_loss_anchor = (
                self.chamfer_weight
                * chamfer_distance(
                    pred_points_anchor,
                    points_anchor,
                )[0].item()
            )
        elif self.point_loss_type == 2:
            # chamfer loss + mse loss
            point_loss_action_mse = mse_criterion(pred_points_action, points_action)
            point_loss_anchor_mse = mse_criterion(pred_points_anchor, points_anchor)

            point_loss_action_chamfer = (
                self.chamfer_weight
                * chamfer_distance(pred_points_action, points_action)[0].item()
            )
            point_loss_anchor_chamfer = (
                self.chamfer_weight
                * chamfer_distance(pred_points_anchor, points_anchor)[0].item()
            )

            point_loss_action = point_loss_action_mse + point_loss_action_chamfer
            point_loss_anchor = point_loss_anchor_mse + point_loss_anchor_chamfer

        point_loss = (
            self.action_weight * point_loss_action
            + self.anchor_weight * point_loss_anchor
        )

        smoothness_loss_action = mse_criterion(
            pred_flow_action,
            induced_flow_action,
        )
        smoothness_loss_anchor = mse_criterion(
            pred_flow_anchor,
            induced_flow_anchor,
        )
        smoothness_loss = (
            self.action_weight * smoothness_loss_action
            + self.anchor_weight * smoothness_loss_anchor
        )

        loss = point_loss + self.smoothness_weight * smoothness_loss

        T_action = pred_T_action
        # log_action = se3_log_map(T_action.get_matrix().detach())
        # trans_mag = log_action[:,:3].norm(dim=-1).sum()
        # rot_mag = log_action[:,3:].abs().sum()

        log_values[loss_prefix + "point_loss_action"] = point_loss_action
        log_values[loss_prefix + "point_loss_anchor"] = point_loss_anchor
        log_values[loss_prefix + "point_loss"] = point_loss
        log_values[loss_prefix + "smoothness_loss_action"] = smoothness_loss_action
        log_values[loss_prefix + "smoothness_loss_anchor"] = smoothness_loss_anchor
        log_values[loss_prefix + "smoothness_loss"] = smoothness_loss

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
        if self.point_loss_type == 2:
            log_values[loss_prefix + "point_loss_action_mse"] = point_loss_action_mse
            log_values[loss_prefix + "point_loss_anchor_mse"] = point_loss_anchor_mse
            log_values[
                loss_prefix + "point_loss_action_chamfer"
            ] = point_loss_action_chamfer
            log_values[
                loss_prefix + "point_loss_anchor_chamfer"
            ] = point_loss_anchor_chamfer

        # log_values[loss_prefix+'trans_mag'] = trans_mag
        # log_values[loss_prefix+'rot_mag'] = rot_mag

        return loss, log_values

    def extract_flow_and_weight(self, x):
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            pred_w = torch.sigmoid(x[:, :, 3])
        else:
            pred_w = None
        return pred_flow, pred_w

    def forward(self, points_actions, points_anchor):
        if self.return_flow_component:
            model_output = self.model(points_actions, points_anchor)
            x_action = model_output["flow_action"]
            x_anchor = model_output["flow_anchor"]
            residual_flow_action = model_output["residual_flow_action"]
            residual_flow_anchor = model_output["residual_flow_anchor"]
            corr_flow_action = model_output["corr_flow_action"]
            corr_flow_anchor = model_output["corr_flow_anchor"]

        else:
            x_action, x_anchor = self.model(points_actions, points_anchor)

        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        pred_T_action = flow2pose(
            points_actions,
            pred_flow_action,
            pred_w_action,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        pred_T_anchor = flow2pose(
            points_anchor,
            pred_flow_anchor,
            pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        pred_T = pred_T_action.compose(pred_T_anchor.inverse())
        return pred_T_action, pred_T_anchor, pred_T  # , pred_T.get_matrix()

    def module_step(self, batch, batch_idx):
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
            x_action, x_anchor = self.model(points_trans_action, points_trans_anchor)

        log_values = {}
        loss, log_values = self.cal_loss(
            x_action, x_anchor, batch, log_values=log_values, loss_prefix=""
        )
        if self.return_flow_component:
            loss_residual, log_values = self.cal_loss(
                residual_flow_action,
                residual_flow_anchor,
                batch,
                log_values=log_values,
                loss_prefix="residual_",
            )
            loss_corr, log_values = self.cal_loss(
                corr_flow_action,
                corr_flow_anchor,
                batch,
                log_values=log_values,
                loss_prefix="corres_",
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
            x_action, x_anchor = self.model(points_trans_action, points_trans_anchor)

        pred_flow_action = x_action[:, :, :3]
        if x_action.shape[2] > 3:
            pred_w_action = torch.sigmoid(x_action[:, :, 3])
        else:
            pred_w_action = None

        pred_flow_anchor = x_anchor[:, :, :3]
        if x_anchor.shape[2] > 3:
            pred_w_anchor = torch.sigmoid(x_anchor[:, :, 3])
        else:
            pred_w_anchor = None

        # Transfrom predicted using both flows.
        # Currently not useing this. Could replace pred_T_action with this.
        pred_flow = torch.cat([pred_flow_action, pred_flow_anchor], dim=1)
        # polarity = torch.cat([torch.ones_like(pred_w_action), torch.zeros_like(pred_w_anchor)])
        # pred_w = torch.cat([pred_w_action, pred_w_anchor])
        # pred_T = dualflow2pose(points_trans, pred_flow, polarity,
        #     weights=pred_w, return_transform3d=True)

        # Transfrom predicted from X_0 to X_1
        pred_T_action = flow2pose(
            points_trans_action,
            pred_flow_action,
            pred_w_action,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        induced_flow_action = (
            pred_T_action.transform_points(points_trans_action) - points_trans_action
        ).detach()
        pred_points_action = pred_T_action.compose(T1.inverse()).transform_points(
            points_trans_action
        )

        # Transfrom predicted from X_1 to X_0
        # Should be pred_T_0^-1
        # Could rewrite the action points s.t. we don't need the inverse
        pred_T_anchor = flow2pose(
            points_trans_anchor,
            pred_flow_anchor,
            pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
        )
        induced_flow_anchor = (
            pred_T_anchor.transform_points(points_trans_anchor) - points_trans_anchor
        ).detach()
        pred_points_anchor = pred_T_anchor.compose(T0.inverse()).transform_points(
            points_trans_anchor
        )
        action_transformed = pred_T_action.transform_points(points_trans_action)
        anchor_transformed = pred_T_anchor.transform_points(points_trans_anchor)

        points = torch.cat([points_action, points_anchor], dim=1)
        points_trans = torch.cat([points_trans_action, points_trans_anchor], dim=1)
        points_target_disp = points[0]  # .detach().cpu().numpy()
        points_trans_disp = points_trans[0]  # .detach().cpu().numpy()
        flow_trans_disp = pred_flow[0]  # .detach().cpu().numpy()

        points_target_disp_action = points_action[0]  # .detach().cpu().numpy()
        # .detach().cpu().numpy()
        points_trans_disp_action = points_trans_action[0]
        flow_trans_disp_action = pred_flow_action[0]  # .detach().cpu().numpy()

        points_target_disp_anchor = points_anchor[0]  # .detach().cpu().numpy()
        # .detach().cpu().numpy()
        points_trans_disp_anchor = points_trans_anchor[0]
        flow_trans_disp_anchor = pred_flow_anchor[0]  # .detach().cpu().numpy()

        index_interval = 10
        res_images = {}
        res_images["all_input_points"] = scatter3d(
            xs=[points_target_disp, points_trans_disp], cs=["b", "r"]
        )
        res_images["input_points_action"] = quiver3d(
            xs=[
                points_target_disp_action[::index_interval],
                points_trans_disp_action[::index_interval],
            ],
            vs=[None, flow_trans_disp_action[::index_interval]],
            cxs=["b", "r"],
            cvs=[None, "m"],
        )
        res_images["input_points_anchor"] = quiver3d(
            xs=[
                points_target_disp_anchor[::index_interval],
                points_trans_disp_anchor[::index_interval],
            ],
            vs=[None, flow_trans_disp_anchor[::index_interval]],
            cxs=["b", "r"],
            cvs=[None, "m"],
        )
        res_images["raw_input_points_action"] = scatter3d(
            xs=[points_target_disp_action, points_trans_disp_action],
            cs=["b", "r"],
        )
        res_images["raw_input_points_anchor"] = scatter3d(
            xs=[points_target_disp_anchor, points_trans_disp_anchor],
            cs=["b", "r"],
        )
        res_images["raw_input_points"] = scatter3d(
            xs=[points_trans_disp_action, points_trans_disp_anchor],
            cs=["b", "r"],
        )
        res_images["demo_points"] = scatter3d(
            xs=[points_target_disp_action, points_target_disp_anchor],
            cs=["b", "r"],
        )

        res_images["demo_points_apply_action_transform"] = scatter3d(
            xs=[action_transformed[0], points_trans_disp_anchor],
            cs=["b", "r"],
        )
        res_images["demo_points_apply_anchor_transform"] = scatter3d(
            xs=[anchor_transformed[0], points_trans_disp_action],
            cs=["b", "r"],
        )

        if self.display_action:
            # flow_trans_disp = classes[0,:,0].unsqueeze(-1).detach().cpu().numpy() * flow_trans_disp

            points_action_disp = points_action[0]  # .detach().cpu().numpy()
            # .detach().cpu().numpy()
            pred_points_action_disp = pred_points_action[0]

            points_trans_anchored = T1.inverse().transform_points(points_trans)
            # .detach().cpu().numpy()
            points_trans_anchored_disp = points_trans_anchored[0]

            action_points_anchored_disp = pred_T_action.compose(
                T1.inverse()
            ).transform_points(points_trans_action)
            # .detach().cpu().numpy()
            action_points_anchored_disp = action_points_anchored_disp[0]

            res_images["loss_points_action"] = scatter3d(
                xs=[points_action_disp, pred_points_action_disp],
                cs=["g", "r"],
            )

            res_images["action_points"] = scatter3d(
                xs=[
                    points_target_disp,
                    points_trans_anchored_disp,
                    action_points_anchored_disp,
                ],
                cs=["b", "r", "m"],
            )
        if self.display_anchor:
            points_anchor_disp = points_anchor[0]  # .detach().cpu().numpy()
            # .detach().cpu().numpy()
            pred_points_anchor_disp = pred_points_anchor[0]

            points_trans_anchored = T0.inverse().transform_points(points_trans)
            # .detach().cpu().numpy()
            points_trans_anchored_disp = points_trans_anchored[0]

            anchor_points_anchored_disp = pred_T_anchor.compose(
                T0.inverse()
            ).transform_points(points_trans_anchor)
            # .detach().cpu().numpy()
            anchor_points_anchored_disp = anchor_points_anchored_disp[0]

            res_images["loss_points_1"] = scatter3d(
                xs=[points_anchor_disp, pred_points_anchor_disp],
                cs=["g", "r"],
            )

            res_images["action_points_1"] = scatter3d(
                xs=[
                    points_target_disp,
                    points_trans_anchored_disp,
                    anchor_points_anchored_disp,
                ],
                cs=["b", "r", "m"],
            )
        return res_images
