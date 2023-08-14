from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset

from taxpose.datasets.base import (
    PlacementPointCloudDataset,
    PlacementPointCloudDatasetConfig,
)
from taxpose.datasets.ndf import ObjectClass
from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion
from taxpose.utils.se3 import random_se3


@dataclass
class PointCloudDatasetConfig:
    demo_dset: PlacementPointCloudDatasetConfig  # Config for actually loading the underlying demo dataset.

    # Config for the dataset wrapper.
    num_points: int = 1024
    rotation_variance: float = np.pi
    translation_variance: float = 0.5
    dataset_size: int = 1000
    angle_degree: int = 180
    synthetic_occlusion: bool = False
    ball_radius: Optional[float] = None
    plane_occlusion: bool = False
    ball_occlusion: bool = False
    plane_standoff: Optional[float] = None
    occlusion_class: ObjectClass = ObjectClass.GRIPPER
    symmetric_class: Optional[int] = None

    # Unused.
    overfit: bool = False
    num_overfit_transforms: int = 3
    gripper_lr_label: bool = False


def make_dataset(
    cfg: PlacementPointCloudDatasetConfig,
) -> PlacementPointCloudDataset:
    if cfg.dataset_type == "ndf":
        import taxpose.datasets.ndf as ndf

        return ndf.NDFPointCloudDataset(cast(ndf.NDFPointCloudDatasetConfig, cfg))
    elif cfg.dataset_type == "rlbench":
        import taxpose.datasets.rlbench as rlbench

        return rlbench.RLBenchPointCloudDataset(
            cast(rlbench.RLBenchPointCloudDatasetConfig, cfg)
        )


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointCloudDatasetConfig):
        self.dataset = make_dataset(cfg.demo_dset)
        self.dataset_size = cfg.dataset_size
        self.num_points = cfg.num_points
        # Path('/home/bokorn/src/ndf_robot/notebooks')
        self.rot_var = cfg.rotation_variance
        self.trans_var = cfg.translation_variance
        self.action_class = cfg.demo_dset.action_class
        self.anchor_class = cfg.demo_dset.anchor_class
        self.symmetric_class = cfg.symmetric_class  # None if no symmetric class exists
        self.angle_degree = cfg.angle_degree

        self.overfit = cfg.overfit
        self.gripper_lr_label = cfg.gripper_lr_label
        self.dataset_indices = cfg.demo_dset.dataset_indices
        self.num_overfit_transforms = cfg.num_overfit_transforms
        self.T0_list = []
        self.T1_list = []
        self.synthetic_occlusion = cfg.synthetic_occlusion
        self.ball_radius = cfg.ball_radius
        self.plane_standoff = cfg.plane_standoff
        self.plane_occlusion = cfg.plane_occlusion
        self.ball_occlusion = cfg.ball_occlusion
        self.occlusion_class = cfg.occlusion_class

    # def get_fixed_transforms(self):
    #     points_action, points_anchor, _ = self.load_data(
    #         self.filenames[0],
    #         action_class=self.action_class,
    #         anchor_class=self.anchor_class,
    #     )
    #     if self.overfit:
    #         # torch.random.manual_seed(0)
    #         for i in range(self.num_overfit_transforms):
    #             a = random_se3(
    #                 1,
    #                 rot_var=self.rot_var,
    #                 trans_var=self.trans_var,
    #                 device=points_action.device,
    #             )
    #             b = random_se3(
    #                 1,
    #                 rot_var=self.rot_var,
    #                 trans_var=self.trans_var,
    #                 device=points_anchor.device,
    #             )
    #             self.T0_list.append(a)
    #             self.T1_list.append(b)
    #     return

    def project_to_xy(self, vector):
        """
        vector: num_poins, 3
        """
        if len(vector.shape) > 1:
            vector[:, -1] = 0
        elif len(vector.shape) == 1:
            vector[-1] = 0
        return vector

    def get_sym_label(
        self, action_cloud, anchor_cloud, action_class, anchor_class, discrete=True
    ):
        assert 0 in [
            action_class,
            anchor_class,
        ], "class 0 must be here somewhere as the manipulation object of interest"
        if action_class == 0:
            sym_breaking_class = action_class
            center_class = anchor_class
            points_sym = action_cloud[0]
            points_nonsym = anchor_cloud[0]
        elif anchor_class == 0:
            sym_breaking_class = anchor_class
            center_class = action_class
            points_sym = anchor_cloud[0]
            points_nonsym = action_cloud[0]

        non_sym_center = points_nonsym.mean(axis=0)
        sym_center = points_sym.mean(axis=0)
        sym2nonsym = non_sym_center - sym_center
        sym2nonsym = self.project_to_xy(sym2nonsym)

        sym_vec = points_sym - sym_center
        sym_vec = self.project_to_xy(sym_vec)
        if discrete:
            sym_cls = torch.sign(torch.matmul(sym_vec, sym2nonsym)).unsqueeze(
                0
            )  # num_points, 1

        return sym_cls

    def __getitem__(self, index):
        data_ix = torch.randint(len(self.dataset), [1])
        data = self.dataset[data_ix]
        points_action = torch.from_numpy(data["points_action"])
        points_anchor = torch.from_numpy(data["points_anchor"])
        action_symmetry_features = torch.from_numpy(data["action_symmetry_features"])
        anchor_symmetry_features = torch.from_numpy(data["anchor_symmetry_features"])
        action_symmetry_rgb = torch.from_numpy(data["action_symmetry_rgb"])
        anchor_symmetry_rgb = torch.from_numpy(data["anchor_symmetry_rgb"])
        # symmetric_cls = torch.from_numpy(data["symmetric_cls"])

        # if self.overfit:
        #     transform_idx = torch.randint(
        #         self.num_overfit_transforms, (1,)).item()
        #     T0 = self.T0_list[transform_idx]
        #     T1 = self.T1_list[transform_idx]
        # else:
        T0 = random_se3(
            1,
            rot_var=self.rot_var,
            trans_var=self.trans_var,
            device=points_action.device,
        )
        T1 = random_se3(
            1,
            rot_var=self.rot_var,
            trans_var=self.trans_var,
            device=points_anchor.device,
        )

        if points_action.shape[1] > self.num_points:
            if self.synthetic_occlusion and self.action_class == self.occlusion_class:
                if self.ball_occlusion:
                    points_action, mask = ball_occlusion(
                        points_action[0], radius=self.ball_radius
                    )
                    points_action = points_action.unsqueeze(0)
                    action_symmetry_features = action_symmetry_features[mask]

                    if action_symmetry_rgb is not None:
                        action_symmetry_rgb = action_symmetry_rgb[mask]

                if self.plane_occlusion:
                    points_action, mask = plane_occlusion(
                        points_action[0], stand_off=self.plane_standoff
                    )
                    points_action = points_action.unsqueeze(0)
                    action_symmetry_features = action_symmetry_features[mask]

                    if action_symmetry_rgb is not None:
                        action_symmetry_rgb = action_symmetry_rgb[mask]

            if points_action.shape[1] > self.num_points:
                points_action, action_ids = sample_farthest_points(
                    points_action, K=self.num_points, random_start_point=True
                )
                action_symmetry_features = action_symmetry_features[
                    0, action_ids.view(-1)
                ][None]

                if action_symmetry_rgb is not None:
                    action_symmetry_rgb = action_symmetry_rgb[0, action_ids.view(-1)][
                        None
                    ]
            elif points_action.shape[1] < self.num_points:
                raise NotImplementedError(
                    f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
                )

            # if len(symmetric_cls) > 0:
            #     symmetric_cls = symmetric_cls[action_ids.view(-1)]
        elif points_action.shape[1] < self.num_points:
            raise NotImplementedError(
                f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
            )

        if points_anchor.shape[1] > self.num_points:
            if self.synthetic_occlusion and self.anchor_class == self.occlusion_class:
                if self.ball_occlusion:
                    points_anchor, mask = ball_occlusion(
                        points_anchor[0], radius=self.ball_radius
                    )
                    points_anchor = points_anchor.unsqueeze(0)
                    anchor_symmetry_features = anchor_symmetry_features[mask]
                    breakpoint()

                    if anchor_symmetry_rgb is not None:
                        anchor_symmetry_rgb = anchor_symmetry_rgb[mask]

                if self.plane_occlusion:
                    points_anchor, mask = plane_occlusion(
                        points_anchor[0], stand_off=self.plane_standoff
                    )
                    points_anchor = points_anchor.unsqueeze(0)
                    anchor_symmetry_features = anchor_symmetry_features[mask]
                    breakpoint()

                    if anchor_symmetry_rgb is not None:
                        anchor_symmetry_rgb = anchor_symmetry_rgb[mask]
            if points_anchor.shape[1] > self.num_points:
                points_anchor_pre = points_anchor
                points_anchor, anchor_ids = sample_farthest_points(
                    points_anchor, K=self.num_points, random_start_point=True
                )
                try:
                    anchor_symmetry_features = anchor_symmetry_features[
                        0, anchor_ids.view(-1)
                    ][None]
                except:
                    breakpoint()
                if anchor_symmetry_rgb is not None:
                    anchor_symmetry_rgb = anchor_symmetry_rgb[0, anchor_ids.view(-1)][
                        None
                    ]
            elif points_anchor.shape[1] < self.num_points:
                raise NotImplementedError(
                    f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})"
                )
        elif points_anchor.shape[1] < self.num_points:
            raise NotImplementedError(
                f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})"
            )

        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)

        # if self.symmetric_class is not None:
        if False:
            symmetric_cls = self.get_sym_label(
                action_cloud=points_action,
                anchor_cloud=points_anchor,
                action_class=self.action_class,
                anchor_class=self.anchor_class,
            )  # num_points, 1
            symmetric_cls = symmetric_cls.unsqueeze(-1)
            if self.action_class == 0:
                points_action = torch.cat([points_action, symmetric_cls], axis=-1)

                points_anchor = torch.cat(
                    [points_anchor, torch.ones(symmetric_cls.shape)], axis=-1
                )
                points_action_trans = torch.cat(
                    [points_action_trans, symmetric_cls], axis=-1
                )
                points_anchor_trans = torch.cat(
                    [points_anchor_trans, torch.ones(symmetric_cls.shape)], axis=-1
                )

            elif self.anchor_class == 0:
                points_anchor = torch.cat([points_anchor, symmetric_cls], axis=-1)

                points_action = torch.cat(
                    [points_action, torch.ones(symmetric_cls.shape)], axis=-1
                )
                points_anchor_trans = torch.cat(
                    [points_anchor_trans, symmetric_cls], axis=-1
                )
                points_action_trans = torch.cat(
                    [points_action_trans, torch.ones(symmetric_cls.shape)], axis=-1
                )

        data = {
            "points_action": points_action.squeeze(0),
            "points_anchor": points_anchor.squeeze(0),
            "points_action_trans": points_action_trans.squeeze(0),
            "points_anchor_trans": points_anchor_trans.squeeze(0),
            "T0": T0.get_matrix().squeeze(0),
            "T1": T1.get_matrix().squeeze(0),
            # "symmetric_cls": symmetric_cls,
            "action_symmetry_features": action_symmetry_features.squeeze(0),
            "anchor_symmetry_features": anchor_symmetry_features.squeeze(0),
            "action_symmetry_rgb": action_symmetry_rgb.squeeze(0),
            "anchor_symmetry_rgb": anchor_symmetry_rgb.squeeze(0),
        }

        return data

    def __len__(self):
        return self.dataset_size
