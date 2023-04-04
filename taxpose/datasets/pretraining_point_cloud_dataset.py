import os
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset


class PretrainingPointCloudDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        dataset_indices=[10],
        cloud_type="final",
        action_class=0,
        num_points=1000,
    ):
        self.dataset_root = Path(dataset_root)
        self.cloud_type = cloud_type
        self.num_points = num_points

        self.action_class = action_class

        self.dataset_indices = dataset_indices
        if self.dataset_indices == "None":
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

            if points_action.shape[1] > self.num_points:
                points_action, action_ids = sample_farthest_points(
                    points_action, K=self.num_points, random_start_point=True
                )

            if points_action.shape[1] < self.num_points:
                bad_demo_id.append(i)
        return bad_demo_id

    def __getitem__(self, index):
        filename = self.filenames[torch.randint(len(self.filenames), [1]).item()]
        points_action = self.load_data(filename, action_class=self.action_class)

        if points_action.shape[1] > self.num_points:
            points_action, action_ids = sample_farthest_points(
                points_action, K=self.num_points, random_start_point=True
            )

        return points_action.squeeze(0)

    def __len__(self):
        return 100
