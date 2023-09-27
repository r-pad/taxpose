import numpy as np
import torch
import wandb
from pytorch3d.transforms import Transform3d

from scripts.evaluate_ndf_mug_standalone import dualflow2pose
from taxpose.datasets.ndf import OBJECT_DEMO_LABELS, ObjectClass
from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.color_utils import get_color
from taxpose.utils.se3 import get_degree_angle, get_translation, pure_translation_se3
from taxpose.utils.symmetry_utils import (
    get_sym_label_pca_grasp,
    get_sym_label_pca_place,
)


class EquivarianceTestingModule(PointCloudTrainingModule):
    """DIFF: adding some return flow..."""

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
        action: str = "grasp",
        object_type: str = "mug",
        action_class: int = 0,
        anchor_class: int = 0,
        normalize_dist: bool = True,
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
        self.action = action
        self.object_type = object_type

        self.action_class = action_class
        self.anchor_class = anchor_class
        self.normalize_dist = normalize_dist

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

    def get_symmetry_labels(self, points_action, points_anchor, skip_symmetry=False):
        if (
            self.object_type in {ObjectClass.BOTTLE, ObjectClass.BOWL}
            and not skip_symmetry
        ):
            if self.action == "grasp":
                sym_dict = get_sym_label_pca_grasp(
                    action_cloud=torch.as_tensor(points_action),
                    anchor_cloud=torch.as_tensor(points_anchor),
                    action_class=self.action_class,
                    anchor_class=self.anchor_class,
                    object_type=self.object_type,
                    normalize_dist=self.normalize_dist,
                )

            elif self.action == "place":
                sym_dict = get_sym_label_pca_place(
                    action_cloud=torch.as_tensor(points_action),
                    anchor_cloud=torch.as_tensor(points_anchor),
                    action_class=self.action_class,
                    anchor_class=self.anchor_class,
                    normalize_dist=self.normalize_dist,
                )

            symmetric_cls = sym_dict["cts_cls"]  # 1, num_points
            symmetric_cls = symmetric_cls.unsqueeze(-1).numpy()  # 1, 1, num_points

            # We want to color the gripper somehow...
            if self.action_class == OBJECT_DEMO_LABELS[ObjectClass.GRIPPER]:
                nonsymmetric_cls = sym_dict["cts_cls_nonsym"]  # 1, num_points
                # 1, 1, num_points
                nonsymmetric_cls = nonsymmetric_cls.unsqueeze(-1).numpy()
            else:
                nonsymmetric_cls = None

            symmetry_xyzrgb = sym_dict["fig"]
            if self.action_class == 0:
                if nonsymmetric_cls is None:
                    nonsymmetric_cls = np.ones(
                        (1, points_anchor.shape[1], 1), dtype=np.float32
                    )
                action_symmetry_features = symmetric_cls
                anchor_symmetry_features = nonsymmetric_cls
            elif self.anchor_class == 0:
                if nonsymmetric_cls is None:
                    nonsymmetric_cls = np.ones(
                        (1, points_action.shape[1], 1), dtype=np.float32
                    )
                action_symmetry_features = nonsymmetric_cls
                anchor_symmetry_features = symmetric_cls
            else:
                raise ValueError("this should not happen")
        else:
            action_symmetry_features = np.ones(
                (1, points_action.shape[1], 1), dtype=np.float32
            )
            anchor_symmetry_features = np.ones(
                (1, points_anchor.shape[1], 1), dtype=np.float32
            )
        return {
            "action_symmetry_features": action_symmetry_features,
            "anchor_symmetry_features": anchor_symmetry_features,
        }

    def get_transform(self, points_trans_action, points_trans_anchor):
        for i in range(self.loop):
            # TODO: fix for bottle bowl.

            # action_symm = torch.ones_like(
            #     points_trans_action[:, :, :1], device=points_trans_action.device
            # )
            # anchor_symm = torch.ones_like(
            #     points_trans_anchor[:, :, :1], device=points_trans_anchor.device
            # )
            symm_feats = self.get_symmetry_labels(
                points_trans_action.cpu().numpy(), points_trans_anchor.cpu().numpy()
            )
            action_symm = torch.as_tensor(symm_feats["action_symmetry_features"]).to(
                points_trans_action.device
            )
            anchor_symm = torch.as_tensor(symm_feats["anchor_symmetry_features"]).to(
                points_trans_anchor.device
            )
            # breakpoint()

            model_output = self.model(
                points_trans_action, points_trans_anchor, action_symm, anchor_symm
            )
            x_action = model_output["flow_action"]
            x_anchor = model_output["flow_anchor"]

            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]

            # If we've applied some sampling, we need to extract the predictions too...
            if "sampled_ixs_action" in model_output:
                ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
                # points_action = torch.take_along_dim(points_action, ixs_action, dim=1)
                points_trans_action = torch.take_along_dim(
                    points_trans_action, ixs_action, dim=1
                )

            if "sampled_ixs_anchor" in model_output:
                ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
                # points_anchor = torch.take_along_dim(points_anchor, ixs_anchor, dim=1)
                points_trans_anchor = torch.take_along_dim(
                    points_trans_anchor, ixs_anchor, dim=1
                )

            ans_dict = self.predict(
                x_action=x_action,
                x_anchor=x_anchor,
                points_trans_action=points_trans_action,
                points_trans_anchor=points_trans_anchor,
            )
            if i == 0:
                pred_T_action = ans_dict["pred_T_action"]

                if self.loop == 1:
                    return ans_dict
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
