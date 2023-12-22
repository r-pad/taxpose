from dataclasses import dataclass
from typing import Dict, Optional, Protocol, TypedDict

import numpy as np
import torch
import torch.nn as nn

from taxpose.datasets.enums import ObjectClass
from taxpose.datasets.ndf import OBJECT_DEMO_LABELS
from taxpose.utils.se3 import dualflow2pose, pure_translation_se3
from taxpose.utils.symmetry_utils import (
    get_sym_label_pca_grasp,
    get_sym_label_pca_place,
)


class PointPredictions(TypedDict):
    flow_action: torch.Tensor
    flow_anchor: Optional[torch.Tensor]


class PointPredictor(Protocol):
    def forward(
        self,
        points_action: torch.Tensor,
        points_anchor: torch.Tensor,
        features_action: Optional[torch.Tensor] = None,  # Symmetry features
        features_anchor: Optional[torch.Tensor] = None,  # Symmetry features
    ) -> PointPredictions:
        ...


@dataclass
class SymmetryConfig:
    """Information about the symmetry of the object."""

    action_class: int
    anchor_class: int
    object_type: ObjectClass
    normalize_dist: bool
    action: str


@dataclass
class TAXPoseReasoningConfig:
    loop: int = 1
    weight_normalize: bool = True
    softmax_temperature: float = 1.0


class TAXPoseReasoning(nn.Module):
    def __init__(
        self, point_pred_module: PointPredictor, cfg: TAXPoseReasoningConfig
    ) -> None:
        super().__init__()

        self.model = point_pred_module

        self.loop = cfg.loop
        self.weight_normalize = cfg.weight_normalize
        self.softmax_temperature = cfg.softmax_temperature

        # Symmetry information.
        # self.object_type = cfg.symmetry_cfg.object_type
        # self.action_class = cfg.symmetry_cfg.action_class
        # self.anchor_class = cfg.symmetry_cfg.anchor_class
        # self.normalize_dist = cfg.symmetry_cfg.normalize_dist
        # self.action = cfg.symmetry_cfg.action

    def forward(
        self,
        points_action,
        points_anchor,
        features_trans_action=None,
        features_trans_anchor=None,
    ) -> Dict:
        return self.get_transform(
            points_action,
            points_anchor,
            features_trans_action,
            features_trans_anchor,
        )

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
        features_trans_action=None,
        features_trans_anchor=None,
    ):
        for i in range(self.loop):
            # symm_feats = self.get_symmetry_labels(
            #     points_trans_action.cpu().numpy(), points_trans_anchor.cpu().numpy()
            # )
            # action_symm = torch.as_tensor(symm_feats["action_symmetry_features"]).to(
            #     points_trans_action.device
            # )
            # anchor_symm = torch.as_tensor(symm_feats["anchor_symmetry_features"]).to(
            #     points_trans_anchor.device
            # )
            # breakpoint()

            model_output = self.model(
                points_trans_action,
                points_trans_anchor,
                features_trans_action,
                features_trans_anchor,
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
                raise ValueError("this should not happen")
                pred_T_action = pred_T_action.compose(
                    T_trans.inverse()
                    .compose(ans_dict["pred_T_action"])
                    .compose(T_trans)
                )
                ans_dict["pred_T_action"] = pred_T_action
            raise ValueError("this should not happen")
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

    def get_symmetry_labels(
        self, points_action, points_anchor, skip_symmetry=False
    ) -> Dict[str, Optional[np.ndarray]]:
        # If there's no symmetry, we don't need to do anything.
        # if self.symmetry_cfg is None:
        #     return {"action_symmetry_features": None, "anchor_symmetry_features": None}

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

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            pred_w = torch.sigmoid(x[:, :, 3])
        else:
            pred_w = None
        return pred_flow, pred_w

    def predict(self, x_action, x_anchor, points_trans_action, points_trans_anchor):
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        # There could be more options for what to choose...

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
