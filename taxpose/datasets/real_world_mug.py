import functools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from taxpose.datasets.base import PlacementPointCloudData


@dataclass
class RealWorldMugPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "real_world_mug"
    dataset_root: Path
    dataset_indices: Optional[List[int]] = None
    start_anchor: bool = False
    min_num_cameras: int = 4
    max_num_cameras: int = 4
    num_points: int = 1024


class RealWorldMugPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: RealWorldMugPointCloudDatasetConfig):
        # Check that all the files in the dataset are there.

        assert cfg.dataset_indices is not None and len(cfg.dataset_indices) > 0

        self.filenames = [
            Path(cfg.dataset_root) / f"{idx}.pkl" for idx in cfg.dataset_indices
        ]

        # Make sure each file exists.
        for filename in self.filenames:
            assert filename.exists()

        self.start_anchor = cfg.start_anchor
        self.min_num_cameras = cfg.min_num_cameras
        self.max_num_cameras = cfg.max_num_cameras
        self.num_points = cfg.num_points

    # @staticmethod
    @functools.cache
    def load_data(self, filename, start_anchor=False):
        with open(filename, "rb") as f:
            sensor_data = pickle.load(f)

        points_action_np = sensor_data["end_action"]
        if start_anchor:
            points_anchor_np = sensor_data["start_anchor"]
        else:
            points_anchor_np = sensor_data["end_anchor"]

        points_action_cls = sensor_data["end_action_cam_label"]
        if start_anchor:
            points_anchor_cls = sensor_data["start_anchor_cam_label"]
        else:
            points_anchor_cls = sensor_data["end_anchor_cam_label"]

        if self.min_num_cameras < 4:
            num_cameras = np.random.randint(
                low=self.min_num_cameras, high=self.max_num_cameras + 1
            )
            sampled_camera_idxs = np.random.choice(4, num_cameras, replace=False)
            action_valid_idxs = np.isin(points_action_cls, sampled_camera_idxs)
            points_action_np = points_action_np[action_valid_idxs]
            anchor_valid_idxs = np.isin(points_anchor_cls, sampled_camera_idxs)
            points_anchor_np = points_anchor_np[anchor_valid_idxs]

        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np
        points_anchor_np = points_anchor_np - points_action_mean_np

        points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
        points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

        if points_action.shape[1] < self.num_points:
            n = self.num_points // points_action.shape[1] + 1
            points_action = torch.cat([points_action] * n, dim=1)
        if points_anchor.shape[1] < self.num_points:
            n = self.num_points // points_anchor.shape[1] + 1
            points_anchor = torch.cat([points_action] * n, dim=1)

        return {
            "points_action": points_action.numpy(),
            "points_anchor": points_anchor.numpy(),
            "action_symmetry_features": np.ones_like(points_action[:, :, :1]),
            "anchor_symmetry_features": np.ones_like(points_anchor[:, :, :1]),
            "action_symmetry_rgb": np.zeros_like(
                points_action[:, :, :3], dtype=np.uint8
            ),
            "anchor_symmetry_rgb": np.zeros_like(
                points_anchor[:, :, :3], dtype=np.uint8
            ),
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.load_data(self.filenames[idx], self.start_anchor)
