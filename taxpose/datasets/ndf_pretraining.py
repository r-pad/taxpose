import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NDFPretrainingPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "ndf_pretraining"
    dataset_root: str
    dataset_indices: list = field(default_factory=lambda: [10])
    cloud_type: str = "final"
    action_class: int = 0  # 0 for mug, 1 for rack, 2 for gripper
    num_points: int = 1024


class NDFPretrainingPointCloudDataset(Dataset):
    def __init__(
        self,
        config: NDFPretrainingPointCloudDatasetConfig,
        # dataset_root,
        # dataset_indices=[10],
        # cloud_type="final",
        # action_class=0,
        # num_points=1000,
    ):
        dataset_root = config.dataset_root
        dataset_indices = config.dataset_indices
        cloud_type = config.cloud_type
        action_class = config.action_class
        num_points = config.num_points

        self.dataset_root = Path(dataset_root)
        self.cloud_type = cloud_type
        self.num_points = num_points

        self.action_class = action_class

        self.dataset_indices = dataset_indices
        if self.dataset_indices == "None" or self.dataset_indices is None:
            dataset_indices = self.get_existing_data_indices()
            self.dataset_indices = dataset_indices
        self.bad_demo_id = self.go_through_list()

        self.filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type}_obj_points.npz"
            for idx in self.dataset_indices
            if idx not in self.bad_demo_id
        ]
        print(len(self.filenames))

    def get_existing_data_indices(self):
        import fnmatch

        num_files = len(
            fnmatch.filter(
                os.listdir(self.dataset_root), f"**_{self.cloud_type}_obj_points.npz"
            )
        )
        file_indices = [
            fn.split("_")[0]
            for fn in fnmatch.filter(
                os.listdir(self.dataset_root), f"**_{self.cloud_type}_obj_points.npz"
            )
        ]
        return file_indices

    # Quite small dataset, so we can cache the data.
    @functools.cache
    def load_data(self, filename, action_class):
        point_data = np.load(filename, allow_pickle=True)
        points_raw_np = point_data["clouds"]
        classes_raw_np = point_data["classes"]

        points_action_np = points_raw_np[classes_raw_np == action_class].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np
        points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)

        return points_action

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
            points_action = self.load_data(
                filename,
                action_class=self.action_class,
            )

            # if points_action.shape[1] > self.num_points:
            #     points_action, action_ids = sample_farthest_points(
            #         points_action, K=self.num_points, random_start_point=True
            #     )

            if points_action.shape[1] < self.num_points:
                bad_demo_id.append(i)
        return bad_demo_id

    def __getitem__(self, index):
        filename = self.filenames[torch.randint(len(self.filenames), [1]).item()]
        points_action = self.load_data(filename, action_class=self.action_class)

        if points_action.shape[1] > self.num_points:
            # points_action, action_ids = sample_farthest_points(
            #     points_action, K=self.num_points, random_start_point=True
            # )
            # Sample random points
            ixs = torch.randperm(points_action.shape[1])[: self.num_points]
            points_action = points_action[:, ixs, :]

        return points_action.squeeze(0)

    def __len__(self):
        return 100
