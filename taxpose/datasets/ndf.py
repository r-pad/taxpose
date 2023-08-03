import fnmatch
import functools
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import ClassVar, List, Optional

import numpy as np
from torch.utils.data import Dataset

from taxpose.datasets.base import PlacementPointCloudData


class ObjectClass(IntEnum):
    MUG = 0
    RACK = 1
    GRIPPER = 2


@dataclass
class NDFPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "ndf"
    dataset_root: Path
    dataset_indices: Optional[List[int]] = None
    num_demo: int = 12
    min_num_points: int = 1024

    cloud_type: str = "teleport"
    action_class: ObjectClass = ObjectClass.MUG
    anchor_class: ObjectClass = ObjectClass.RACK
    min_num_cameras: int = 4
    max_num_cameras: int = 4


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

        if self.dataset_indices is None or self.dataset_indices == "None":
            dataset_indices = self.get_existing_data_indices()
            self.dataset_indices = dataset_indices

        self.bad_demo_id = self.go_through_list()

        self.filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type}_obj_points.npz"
            for idx in self.dataset_indices
            if idx not in self.bad_demo_id
        ]

        if self.num_demo is not None:
            self.filenames = self.filenames[: self.num_demo]

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
            points_action, points_anchor, _ = self.load_data(
                filename,
                action_class=self.action_class,
                anchor_class=self.anchor_class,
            )
            if (points_action.shape[1] < self.min_num_points) or (
                points_anchor.shape[1] < self.min_num_points
            ):
                bad_demo_id.append(i)

        return bad_demo_id

    @functools.cache
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

        # Not really sure what this is...
        symmetric_cls = np.asarray([], dtype=np.float32)

        return points_action, points_anchor, symmetric_cls

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        filename = self.filenames[index]

        points_action, points_anchor, symmetric_cls = self.load_data(
            filename, action_class=self.action_class, anchor_class=self.anchor_class
        )

        return {
            "points_action": points_action,
            "points_anchor": points_anchor,
            "symmetric_cls": symmetric_cls,
        }

    def __len__(self) -> int:
        return len(self.filenames)
