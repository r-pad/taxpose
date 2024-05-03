from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset

from taxpose.datasets.base import (
    PlacementPointCloudDataset,
    PlacementPointCloudDatasetConfig,
)
from taxpose.utils.se3 import random_se3


@dataclass
class PointCloudDatasetConfig:
    demo_dset: PlacementPointCloudDatasetConfig  # Config for actually loading the underlying demo dataset.

    # Config for doing the data augmentations.
    action_rotation_variance: float = np.pi
    anchor_rotation_variance: float = np.pi
    translation_variance: float = 0.5
    action_rot_sample_method: str = "axis_angle"
    anchor_rot_sample_method: str = "axis_angle"
    dataset_size: int = 1000


def make_dataset(
    cfg: PlacementPointCloudDatasetConfig,
) -> PlacementPointCloudDataset:
    if cfg.dataset_type == "ndf":
        import taxpose.datasets.ndf as ndf

        return ndf.NDFPointCloudDataset(cast(ndf.NDFPointCloudDatasetConfig, cfg))
    else:
        raise NotImplementedError(f"Unknown dataset type: {cfg.dataset_type}")


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointCloudDatasetConfig):
        self.dataset = make_dataset(cfg.demo_dset)
        self.dataset_size = cfg.dataset_size
        self.action_rot_var = cfg.action_rotation_variance
        self.anchor_rot_var = cfg.anchor_rotation_variance
        self.trans_var = cfg.translation_variance
        self.action_rot_sample_method = cfg.action_rot_sample_method
        self.anchor_rot_sample_method = cfg.anchor_rot_sample_method

    def __getitem__(self, index):
        data_ix = torch.randint(len(self.dataset), [1]).item()
        data = self.dataset[data_ix]
        points_action = torch.from_numpy(data["points_action"])
        points_anchor = torch.from_numpy(data["points_anchor"])
        action_sym_feats = (
            torch.from_numpy(data["action_symmetry_features"])
            if data["action_symmetry_features"] is not None
            else None
        )
        anchor_sym_feats = (
            torch.from_numpy(data["anchor_symmetry_features"])
            if data["anchor_symmetry_features"] is not None
            else None
        )
        action_sym_rgb = (
            torch.from_numpy(data["action_symmetry_rgb"])
            if data["action_symmetry_rgb"] is not None
            else None
        )
        anchor_sym_rgb = (
            torch.from_numpy(data["anchor_symmetry_rgb"])
            if data["anchor_symmetry_rgb"] is not None
            else None
        )

        T0 = random_se3(
            1,
            rot_var=self.action_rot_var,
            trans_var=self.trans_var,
            device=points_action.device,
            rot_sample_method=self.action_rot_sample_method,
        )
        T1 = random_se3(
            1,
            rot_var=self.anchor_rot_var,
            trans_var=self.trans_var,
            device=points_anchor.device,
            rot_sample_method=self.anchor_rot_sample_method,
        )

        # Transform the points!
        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)

        out_dict = {
            "points_action": points_action.squeeze(0),
            "points_anchor": points_anchor.squeeze(0),
            "points_action_trans": points_action_trans.squeeze(0),
            "points_anchor_trans": points_anchor_trans.squeeze(0),
            "T0": T0.get_matrix().squeeze(0),
            "T1": T1.get_matrix().squeeze(0),
        }
        if action_sym_feats is not None:
            out_dict["action_symmetry_features"] = action_sym_feats.squeeze(0)
            out_dict["anchor_symmetry_features"] = anchor_sym_feats.squeeze(0)
            out_dict["action_symmetry_rgb"] = action_sym_rgb.squeeze(0)
            out_dict["anchor_symmetry_rgb"] = anchor_sym_rgb.squeeze(0)

        if "phase_onehot" in data:
            out_dict["phase_onehot"] = data["phase_onehot"]

        return out_dict

    def __len__(self):
        return self.dataset_size
