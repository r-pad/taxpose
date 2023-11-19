from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset

from taxpose.datasets.base import (
    PlacementPointCloudDataset,
    PlacementPointCloudDatasetConfig,
)
from taxpose.datasets.ndf import (  # compute_demo_symmetry_features,
    OBJECT_LABELS_TO_CLASS,
    ObjectClass,
)
from taxpose.datasets.symmetry_utils import (
    gripper_symmetry_labels,
    nonsymmetric_labels,
    rotational_symmetry_labels,
    scalars_to_rgb,
)
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
    testing_symmetry: bool = False

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
    elif cfg.dataset_type == "real_world_mug":
        import taxpose.datasets.real_world_mug as real_world_mug

        return real_world_mug.RealWorldMugPointCloudDataset(
            cast(real_world_mug.RealWorldMugPointCloudDatasetConfig, cfg)
        )


def compute_demo_symmetry_features(
    points_action: npt.NDArray[np.float32],
    points_anchor: npt.NDArray[np.float32],
    action_class: ObjectClass,
    anchor_class: ObjectClass,
):
    assert len(points_action.shape) == 2
    assert len(points_anchor.shape) == 2
    assert points_action.shape[1] == 3
    assert points_anchor.shape[1] == 3

    if anchor_class == ObjectClass.GRIPPER:
        raise ValueError("Anchor class cannot be the gripper.")

    action_centroid = points_action.mean(axis=0)
    anchor_centroid = points_anchor.mean(axis=0)

    if action_class == ObjectClass.GRIPPER:
        action_sym_feats, _, _ = gripper_symmetry_labels(points_action)
    elif action_class in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
        action_sym_feats, _, _, _ = rotational_symmetry_labels(
            points_action, action_class, anchor_centroid
        )
    else:
        action_sym_feats, _ = nonsymmetric_labels(points_action)

    if anchor_class in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
        anchor_sym_feats, _, _, _ = rotational_symmetry_labels(
            points_anchor, anchor_class, action_centroid
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
        self.num_overfit_transforms = cfg.num_overfit_transforms
        self.T0_list = []
        self.T1_list = []
        self.synthetic_occlusion = cfg.synthetic_occlusion
        self.ball_radius = cfg.ball_radius
        self.plane_standoff = cfg.plane_standoff
        self.plane_occlusion = cfg.plane_occlusion
        self.ball_occlusion = cfg.ball_occlusion
        self.occlusion_class = cfg.occlusion_class
        self.testing_symmetry = cfg.testing_symmetry

        self.demo_dset_cfg = cfg.demo_dset

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
        data_ix = torch.randint(len(self.dataset), [1]).item()
        data = self.dataset[data_ix]
        points_action = torch.from_numpy(data["points_action"])
        points_anchor = torch.from_numpy(data["points_anchor"])
        # action_sym_feats = data["action_symmetry_features"]
        # anchor_sym_feats = data["anchor_symmetry_features"]
        # action_sym_rgb = data["action_symmetry_rgb"]
        # anchor_symmetry_rgb = data["anchor_symmetry_rgb"]

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

        # if action_sym_feats is None or anchor_sym_feats is None:
        #     (
        #         action_sym_feats,
        #         anchor_sym_feats,
        #         action_sym_rgb,
        #         anchor_symmetry_rgb,
        #     ) = compute_symmetry_features(
        #         points_action_trans,
        #         points_anchor_trans,
        #         self.dataset.object_type,
        #         self.dataset.action,
        #         self.action_class,
        #         self.anchor_class,
        #         self.dataset.normalize_dist,
        #         skip_symmetry=False,
        #     )

        # action_sym_feats = torch.from_numpy(action_sym_feats)
        # anchor_sym_feats = torch.from_numpy(anchor_sym_feats)
        # action_sym_rgb = torch.from_numpy(action_sym_rgb)
        # anchor_symmetry_rgb = torch.from_numpy(anchor_symmetry_rgb)

        # Occlusion happens before symmetry.
        def apply_occlusion(points, obj_class):
            if self.synthetic_occlusion and obj_class == self.occlusion_class:
                if self.ball_occlusion:
                    if np.random.rand() > 0.5:
                        points_new, _ = ball_occlusion(
                            points[0], radius=self.ball_radius
                        )

                        # Ignore the occlusion if it's going to mess us up later...
                        if points_new.shape[0] > self.num_points:
                            points = points_new.unsqueeze(0)

                if self.plane_occlusion:
                    if np.random.rand() > 0.5:
                        points_new, _ = plane_occlusion(
                            points[0], stand_off=self.plane_standoff
                        )
                        # Ignore the occlusion if it's going to mess us up later...
                        if points_new.shape[0] > self.num_points:
                            points = points_new.unsqueeze(0)
            return points

        # Apply occlusions.
        if points_action.shape[1] > self.num_points:
            # Apply any occlusions.
            points_action = apply_occlusion(points_action, self.action_class)
        else:
            raise NotImplementedError(
                f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
            )
        if points_anchor.shape[1] > self.num_points:
            points_anchor = apply_occlusion(points_anchor, self.anchor_class)
        else:
            raise NotImplementedError(
                f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})"
            )

        # print(f"Action symmetry features: {action_sym_feats.shape}")
        # print(f"Anchor symmetry features: {anchor_sym_feats.shape}")
        # print(f"Action symmetry rgb: {action_sym_rgb.shape}")
        # print(f"Anchor symmetry rgb: {anchor_sym_rgb.shape}")

        # Downsample after symmetry features.
        def apply_sampling(points):
            points, ids = sample_farthest_points(
                points, K=self.num_points, random_start_point=True
            )

            return points

        if points_action.shape[1] > self.num_points:
            points_action = apply_sampling(points_action)
        else:
            raise NotImplementedError(
                f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
            )

        if points_anchor.shape[1] > self.num_points:
            points_anchor = apply_sampling(points_anchor)
        else:
            raise NotImplementedError(
                f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})"
            )

        # Create symmetry features. This happens after downsampling!!! (it could happen before, but it's not efficient)
        # The core idea here is that we want those symmetry features to be semantically meaningful during
        # demonstrations, so we need to compute them before transforming everything.
        # At test time, there should be a DIFFERENT PROCEDURE!!!
        if self.demo_dset_cfg.dataset_type == "ndf":
            ot = self.dataset.object_type
            action_class = OBJECT_LABELS_TO_CLASS[(ot, self.action_class)]
            anchor_class = OBJECT_LABELS_TO_CLASS[(ot, self.anchor_class)]
            # print(f"Action class: {action_class}, Anchor class: {anchor_class}")

            if self.testing_symmetry:
                (
                    action_sym_feats,
                    anchor_sym_feats,
                    action_sym_rgb,
                    anchor_sym_rgb,
                ) = compute_inference_symmetry_features(
                    points_action[0].cpu().numpy(),
                    points_anchor[0].cpu().numpy(),
                    action_class,
                    anchor_class,
                )
            else:
                (
                    action_sym_feats,
                    anchor_sym_feats,
                    action_sym_rgb,
                    anchor_sym_rgb,
                ) = compute_demo_symmetry_features(
                    points_action[0].cpu().numpy(),
                    points_anchor[0].cpu().numpy(),
                    action_class,
                    anchor_class,
                )

        else:
            # We'll just compute nonsymmetric labels for now.
            (
                action_sym_feats,
                anchor_sym_feats,
                action_sym_rgb,
                anchor_sym_rgb,
            ) = compute_demo_symmetry_features(
                points_action[0].cpu().numpy(),
                points_anchor[0].cpu().numpy(),
                action_class=None,
                anchor_class=None,
            )

        action_sym_feats = torch.from_numpy(action_sym_feats).float()
        anchor_sym_feats = torch.from_numpy(anchor_sym_feats).float()
        action_sym_rgb = torch.from_numpy(action_sym_rgb)
        anchor_sym_rgb = torch.from_numpy(anchor_sym_rgb)

        # Transform the points!
        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)

        data = {
            "points_action": points_action.squeeze(0),
            "points_anchor": points_anchor.squeeze(0),
            "points_action_trans": points_action_trans.squeeze(0),
            "points_anchor_trans": points_anchor_trans.squeeze(0),
            "T0": T0.get_matrix().squeeze(0),
            "T1": T1.get_matrix().squeeze(0),
            "action_symmetry_features": action_sym_feats.squeeze(0),
            "anchor_symmetry_features": anchor_sym_feats.squeeze(0),
            "action_symmetry_rgb": action_sym_rgb.squeeze(0),
            "anchor_symmetry_rgb": anchor_sym_rgb.squeeze(0),
        }

        return data

    def __len__(self):
        return self.dataset_size
