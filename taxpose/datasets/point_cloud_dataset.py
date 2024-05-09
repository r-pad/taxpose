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

    # Handle the synthesizing of various features.
    include_symmetry_features: bool = False


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


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointCloudDatasetConfig):
        self.dataset = make_dataset(cfg.demo_dset)
        self.dataset_size = cfg.dataset_size
        self.action_rot_var = cfg.action_rotation_variance
        self.anchor_rot_var = cfg.anchor_rotation_variance
        self.trans_var = cfg.translation_variance
        self.action_rot_sample_method = cfg.action_rot_sample_method
        self.anchor_rot_sample_method = cfg.anchor_rot_sample_method
        self.cfg = cfg

    def __getitem__(self, index):
        data_ix = torch.randint(len(self.dataset), [1]).item()
        data = self.dataset[data_ix]
        points_action = torch.from_numpy(data["points_action"])
        points_anchor = torch.from_numpy(data["points_anchor"])

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

        # Handle the extra features.

        # We might have different features to include here.
        action_features = None
        anchor_features = None
        if self.cfg.include_symmetry_features:
            if (
                "action_symmetry_features" not in data
                or "anchor_symmetry_features" not in data
                or "action_symmetry_rgb" not in data
                or "anchor_symmetry_rgb" not in data
            ):
                raise ValueError("expected symmetry features to be present in dataset")
            action_sym_feats = torch.from_numpy(
                data["action_symmetry_features"]
            ).squeeze(0)
            anchor_sym_feats = torch.from_numpy(
                data["anchor_symmetry_features"]
            ).squeeze(0)
            action_sym_rgb = torch.from_numpy(data["action_symmetry_rgb"]).squeeze(0)
            anchor_sym_rgb = torch.from_numpy(data["anchor_symmetry_rgb"]).squeeze(0)

            out_dict["action_symmetry_features"] = action_sym_feats
            out_dict["anchor_symmetry_features"] = anchor_sym_feats
            out_dict["action_symmetry_rgb"] = action_sym_rgb
            out_dict["anchor_symmetry_rgb"] = anchor_sym_rgb

            action_features = action_sym_feats
            anchor_features = anchor_sym_feats

        if action_features is not None and anchor_features is not None:
            out_dict["action_features"] = action_features
            out_dict["anchor_features"] = anchor_features

        return out_dict

    def __len__(self):
        return self.dataset_size
