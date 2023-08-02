import fnmatch
import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset

from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion
from taxpose.utils.se3 import random_se3


class PlacementPointCloudData(TypedDict):
    points_action: npt.NDArray[np.float32]  # (1, num_points, 3)
    points_anchor: npt.NDArray[np.float32]  # (1, num_points, 3)
    symmetric_cls: npt.NDArray[np.float32]  # Not really sure what this is...


@dataclass
class NDFPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "ndf"
    dataset_root: Path
    dataset_indices: Optional[List[int]] = None
    num_demo: int = 12
    min_num_points: int = 1024

    cloud_type: str = "final"
    action_class: int = 0
    anchor_class: int = 1
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


@dataclass
class PointClassDatasetConfig:
    demo_dset: NDFPointCloudDataset  # Config for actually loading the underlying demo dataset.

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
    occlusion_class: int = 2
    symmetric_class: Optional[int] = None

    # Unused.
    overfit: bool = False
    num_overfit_transforms: int = 3
    gripper_lr_label: bool = False


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointClassDatasetConfig):
        self.dataset = NDFPointCloudDataset(cfg.demo_dset)
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
        symmetric_cls = torch.from_numpy(data["symmetric_cls"])

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
                    points_action = ball_occlusion(
                        points_action[0], radius=self.ball_radius
                    ).unsqueeze(0)
                if self.plane_occlusion:
                    points_action = plane_occlusion(
                        points_action[0], stand_off=self.plane_standoff
                    ).unsqueeze(0)

            if points_action.shape[1] > self.num_points:
                points_action, action_ids = sample_farthest_points(
                    points_action, K=self.num_points, random_start_point=True
                )
            elif points_action.shape[1] < self.num_points:
                raise NotImplementedError(
                    f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
                )

            if len(symmetric_cls) > 0:
                symmetric_cls = symmetric_cls[action_ids.view(-1)]
        elif points_action.shape[1] < self.num_points:
            raise NotImplementedError(
                f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
            )

        if points_anchor.shape[1] > self.num_points:
            if self.synthetic_occlusion and self.anchor_class == self.occlusion_class:
                if self.ball_occlusion:
                    points_anchor = ball_occlusion(
                        points_anchor[0], radius=self.ball_radius
                    ).unsqueeze(0)
                if self.plane_occlusion:
                    points_anchor = plane_occlusion(
                        points_anchor[0], stand_off=self.plane_standoff
                    ).unsqueeze(0)
            if points_anchor.shape[1] > self.num_points:
                points_anchor, _ = sample_farthest_points(
                    points_anchor, K=self.num_points, random_start_point=True
                )
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

        if self.symmetric_class is not None:
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
            "symmetric_cls": symmetric_cls,
        }

        return data

    def __len__(self):
        return self.dataset_size
