import fnmatch
import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from taxpose.datasets.augmentations import (
    OcclusionConfig,
    maybe_downsample,
    occlusion_fn,
)
from taxpose.datasets.base import PlacementPointCloudData
from taxpose.datasets.enums import ObjectClass, Phase
from taxpose.datasets.env_mod_utils import get_random_distractor_demo  # type: ignore
from taxpose.datasets.symmetry_utils import (
    compute_demo_symmetry_features as new_compute_demo_symmetry_features,
)
from taxpose.utils.symmetry_utils import (
    get_sym_label_pca_grasp,
    get_sym_label_pca_place,
)

#  0 for mug, 1 for rack, 2 for gripper
# These are the labels used in the NDF dataset
# inside demos.
OBJECT_DEMO_LABELS: Dict[ObjectClass, int] = {
    ObjectClass.MUG: 0,
    ObjectClass.RACK: 1,
    ObjectClass.GRIPPER: 2,
    ObjectClass.BOTTLE: 0,
    ObjectClass.BOWL: 0,
    ObjectClass.SLAB: 1,
}

TASK_CLASSES = Literal[ObjectClass.MUG, ObjectClass.BOTTLE, ObjectClass.BOWL]

OBJECT_LABELS_TO_CLASS: Dict[Tuple[TASK_CLASSES, int], ObjectClass] = {
    (ObjectClass.MUG, 0): ObjectClass.MUG,
    (ObjectClass.MUG, 1): ObjectClass.RACK,
    (ObjectClass.MUG, 2): ObjectClass.GRIPPER,
    (ObjectClass.BOTTLE, 0): ObjectClass.BOTTLE,
    (ObjectClass.BOTTLE, 1): ObjectClass.SLAB,
    (ObjectClass.BOTTLE, 2): ObjectClass.GRIPPER,
    (ObjectClass.BOWL, 0): ObjectClass.BOWL,
    (ObjectClass.BOWL, 1): ObjectClass.SLAB,
    (ObjectClass.BOWL, 2): ObjectClass.GRIPPER,
}


@dataclass
class NDFPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "ndf"
    dataset_root: Path
    dataset_indices: Optional[List[int]] = None
    num_demo: int = 12
    min_num_points: int = 1024

    cloud_type: str = "teleport"
    action_class: int = OBJECT_DEMO_LABELS[ObjectClass.MUG]
    anchor_class: int = OBJECT_DEMO_LABELS[ObjectClass.RACK]
    min_num_cameras: int = 4
    max_num_cameras: int = 4

    # Symmetry parameters.
    normalize_dist: bool = True
    object_type: ObjectClass = ObjectClass.MUG
    action: Phase = Phase.GRASP
    symmetry_after_transform: bool = False

    # Augmentation parameters.
    occlusion_cfg: Optional[OcclusionConfig] = None
    distractor_anchor_aug: bool = False
    distractor_rot_sample_method: str = "axis_angle"
    multimodal_transform_base: bool = False


def compute_demo_symmetry_features(
    points_action,
    points_anchor,
    object_type,
    action,
    action_class,
    anchor_class,
    normalize_dist,
    skip_symmetry=False,
):
    # print(
    #     f"object_type: {object_type}, action: {action}, action_class: {action_class}, anchor_class: {anchor_class}"
    # )
    # Handle symmetry.
    if skip_symmetry:
        return None, None, None, None

    if object_type in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
        if action == "grasp":
            sym_dict = get_sym_label_pca_grasp(
                action_cloud=torch.as_tensor(points_action),
                anchor_cloud=torch.as_tensor(points_anchor),
                action_class=action_class,
                anchor_class=anchor_class,
                object_type=object_type,
                normalize_dist=normalize_dist,
            )

        elif action == "place":
            sym_dict = get_sym_label_pca_place(
                action_cloud=torch.as_tensor(points_action),
                anchor_cloud=torch.as_tensor(points_anchor),
                action_class=action_class,
                anchor_class=anchor_class,
                normalize_dist=normalize_dist,
            )

        symmetric_cls = sym_dict["cts_cls"]  # 1, num_points
        symmetric_cls = symmetric_cls.unsqueeze(-1).numpy()  # 1, 1, num_points

        # We want to color the gripper somehow...
        if action_class == OBJECT_DEMO_LABELS[ObjectClass.GRIPPER]:
            nonsymmetric_cls = sym_dict["cts_cls_nonsym"]  # 1, num_points
            # 1, 1, num_points
            nonsymmetric_cls = nonsymmetric_cls.unsqueeze(-1).numpy()
        else:
            nonsymmetric_cls = None

        symmetry_xyzrgb = sym_dict["fig"]
        if action_class == 0:
            if nonsymmetric_cls is None:
                nonsymmetric_cls = np.ones(
                    (1, points_anchor.shape[1], 1), dtype=np.float32
                )
            action_symmetry_features = symmetric_cls
            anchor_symmetry_features = nonsymmetric_cls
            action_symmetry_rgb = symmetry_xyzrgb[: points_action.shape[1], 3:][None]
            anchor_symmetry_rgb = symmetry_xyzrgb[points_action.shape[1] :, 3:][None]
        elif anchor_class == 0:
            if nonsymmetric_cls is None:
                nonsymmetric_cls = np.ones(
                    (1, points_action.shape[1], 1), dtype=np.float32
                )
            action_symmetry_features = nonsymmetric_cls
            anchor_symmetry_features = symmetric_cls
            action_symmetry_rgb = symmetry_xyzrgb[points_anchor.shape[1] :, 3:][None]
            anchor_symmetry_rgb = symmetry_xyzrgb[: points_anchor.shape[1], 3:][None]
        else:
            raise ValueError("this should not happen")
    else:
        action_symmetry_features = np.ones(
            (1, points_action.shape[1], 1), dtype=np.float32
        )
        anchor_symmetry_features = np.ones(
            (1, points_anchor.shape[1], 1), dtype=np.float32
        )
        action_symmetry_rgb = np.zeros((1, points_action.shape[1], 3), dtype=np.uint8)
        anchor_symmetry_rgb = np.zeros((1, points_anchor.shape[1], 3), dtype=np.uint8)

    return (
        action_symmetry_features,
        anchor_symmetry_features,
        action_symmetry_rgb,
        anchor_symmetry_rgb,
    )


class NDFPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: NDFPointCloudDatasetConfig):
        self.dataset_root = Path(cfg.dataset_root)
        self.dataset_indices = cfg.dataset_indices
        self.min_num_points = cfg.min_num_points
        self.num_demo = cfg.num_demo
        self.cloud_type = cfg.cloud_type
        self.action_class = cfg.action_class
        self.anchor_class = cfg.anchor_class
        self.min_num_cameras = cfg.min_num_cameras
        self.max_num_cameras = cfg.max_num_cameras
        self.normalize_dist = cfg.normalize_dist
        self.object_type = cfg.object_type
        self.action = cfg.action
        self.skip_symmetry = cfg.symmetry_after_transform
        self.distractor_anchor_aug = cfg.distractor_anchor_aug
        self.distractor_rot_sample_method = cfg.distractor_rot_sample_method
        self.multimodal_transform_base = cfg.multimodal_transform_base

        if self.dataset_indices is None or self.dataset_indices == "None":
            dataset_indices = self.get_existing_data_indices()
            self.dataset_indices = dataset_indices

        self.bad_demo_id = self.go_through_list()

        self.filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type}_obj_points.npz"
            for idx in self.dataset_indices  # type: ignore
            if idx not in self.bad_demo_id
        ]

        if self.num_demo is not None:
            self.filenames = self.filenames[: self.num_demo]

        self.occlusion_cfg = cfg.occlusion_cfg
        self.occlusion_fn = occlusion_fn(cfg.occlusion_cfg)

    def get_existing_data_indices(self):
        num_files = len(
            fnmatch.filter(
                os.listdir(self.dataset_root), f"**_{self.cloud_type}_obj_points.npz"
            )
        )
        file_indices = [
            int(fn.split("_")[0])
            for fn in fnmatch.filter(
                os.listdir(self.dataset_root), f"**_{self.cloud_type}_obj_points.npz"
            )
        ]
        return file_indices

    def go_through_list(self):
        bad_demo_id = []
        filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type}_obj_points.npz"
            for idx in self.dataset_indices
        ]
        for i in range(len(filenames)):
            filename = filenames[i]
            if i == 0:
                print(filename)
            if not os.path.exists(filename):
                bad_demo_id.append(i)
                continue
            points_action, points_anchor = self.load_data(
                filename,
                action_class=self.action_class,
                anchor_class=self.anchor_class,
            )
            if (points_action.shape[1] < self.min_num_points) or (
                points_anchor.shape[1] < self.min_num_points
            ):
                bad_demo_id.append(i)

        return bad_demo_id

    @functools.lru_cache(maxsize=100)
    def load_data(self, filename, action_class, anchor_class):
        point_data = np.load(filename, allow_pickle=True)
        points_raw_np = point_data["clouds"]
        classes_raw_np = point_data["classes"]
        if self.min_num_cameras < 4:
            camera_idxs = np.concatenate(
                [[0], np.cumsum((np.diff(classes_raw_np) == -2))]
            )
            if not np.all(np.isin(np.arange(4), np.unique(camera_idxs))):
                raise ValueError(
                    "\033[93m"
                    + f"{filename} did not contain all classes in all cameras"
                    + "\033[0m"
                )

            num_cameras = np.random.randint(
                low=self.min_num_cameras, high=self.max_num_cameras + 1
            )
            sampled_camera_idxs = np.random.choice(4, num_cameras, replace=False)
            valid_idxs = np.isin(camera_idxs, sampled_camera_idxs)
            points_raw_np = points_raw_np[valid_idxs]
            classes_raw_np = classes_raw_np[valid_idxs]

        points_action_np = points_raw_np[classes_raw_np == action_class].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
        points_anchor_np = points_anchor_np - points_action_mean_np
        points_anchor_mean_np = points_anchor_np.mean(axis=0)

        points_action = points_action_np.astype(np.float32)[None, ...]
        points_anchor = points_anchor_np.astype(np.float32)[None, ...]

        return (points_action, points_anchor)

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        filename = self.filenames[index]

        (points_action, points_anchor) = self.load_data(
            filename,
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )

        if self.distractor_anchor_aug:
            (
                _,
                points_action,
                points_anchor1,
                points_anchor2,
                debug,
            ) = get_random_distractor_demo(
                None,
                torch.from_numpy(points_action),
                torch.from_numpy(points_anchor),
                transform_base=self.multimodal_transform_base,
                rot_sample_method=self.distractor_rot_sample_method,
            )
            points_action = points_action.numpy()
            points_anchor = torch.cat([points_anchor1, points_anchor2], dim=1).numpy()

        # Apply occlusions
        if self.occlusion_cfg is not None:
            points_action = self.occlusion_fn(
                points_action, self.action_class, self.min_num_points
            )
            points_anchor = self.occlusion_fn(
                points_anchor, self.anchor_class, self.min_num_points
            )

        # Downsample
        points_action = maybe_downsample(points_action, self.min_num_points)
        points_anchor = maybe_downsample(points_anchor, self.min_num_points)

        # Symmetry
        (
            action_symmetry_features,
            anchor_symmetry_features,
            action_symmetry_rgb,
            anchor_symmetry_rgb,
        ) = new_compute_demo_symmetry_features(
            points_action[0],
            points_anchor[0],
            OBJECT_LABELS_TO_CLASS[(self.object_type, self.action_class)],  # type: ignore
            OBJECT_LABELS_TO_CLASS[(self.object_type, self.anchor_class)],  # type: ignore
        )

        assert not isinstance(action_symmetry_features, torch.Tensor)
        assert not isinstance(anchor_symmetry_features, torch.Tensor)

        if action_symmetry_features is not None:
            action_symmetry_features = np.expand_dims(action_symmetry_features, 0)
            anchor_symmetry_features = np.expand_dims(anchor_symmetry_features, 0)
            action_symmetry_rgb = np.expand_dims(action_symmetry_rgb, 0)
            anchor_symmetry_rgb = np.expand_dims(anchor_symmetry_rgb, 0)

        return {
            "points_action": points_action,
            "points_anchor": points_anchor,
            "action_symmetry_features": action_symmetry_features,
            "anchor_symmetry_features": anchor_symmetry_features,
            "action_symmetry_rgb": action_symmetry_rgb,
            "anchor_symmetry_rgb": anchor_symmetry_rgb,
        }

    def __len__(self) -> int:
        return len(self.filenames)
