import os
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset

from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion
from taxpose.utils.se3 import random_se3


class PointCloudDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        dataset_indices=[10],
        cloud_type="final",
        action_class=0,
        anchor_class=1,
        num_points=1024,
        dataset_size=1000,
        rotation_variance=np.pi,
        translation_variance=0.5,
        symmetric_class=None,
        angle_degree=180,
        overfit=False,
        num_overfit_transforms=3,
        gripper_lr_label=False,
        synthetic_occlusion=False,
        ball_radius=None,
        plane_occlusion=False,
        ball_occlusion=False,
        plane_standoff=None,
        occlusion_class=2,
        num_demo=12,
        min_num_cameras=4,
        max_num_cameras=4,
    ):
        self.dataset_size = dataset_size
        self.num_points = num_points
        # Path('/home/bokorn/src/ndf_robot/notebooks')
        self.dataset_root = Path(dataset_root)
        self.cloud_type = cloud_type
        self.rot_var = rotation_variance
        self.trans_var = translation_variance
        self.action_class = action_class
        self.anchor_class = anchor_class
        self.symmetric_class = symmetric_class  # None if no symmetric class exists
        self.angle_degree = angle_degree
        self.min_num_cameras = min_num_cameras
        self.max_num_cameras = max_num_cameras

        self.overfit = overfit
        self.gripper_lr_label = gripper_lr_label
        self.dataset_indices = dataset_indices
        self.num_overfit_transforms = num_overfit_transforms
        self.T0_list = []
        self.T1_list = []
        if self.dataset_indices == "None":
            dataset_indices = self.get_existing_data_indices()
            self.dataset_indices = dataset_indices
        self.bad_demo_id = self.go_through_list()
        self.synthetic_occlusion = synthetic_occlusion
        self.ball_radius = ball_radius
        self.plane_standoff = plane_standoff
        self.plane_occlusion = plane_occlusion
        self.ball_occlusion = ball_occlusion
        self.occlusion_class = occlusion_class
        self.num_demo = num_demo

        self.filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type}_obj_points.npz"
            for idx in self.dataset_indices
            if idx not in self.bad_demo_id
        ]
        if self.num_demo is not None:
            self.filenames = self.filenames[: self.num_demo]

    def get_fixed_transforms(self):
        points_action, points_anchor, _ = self.load_data(
            self.filenames[0],
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )
        if self.overfit:
            # torch.random.manual_seed(0)
            for i in range(self.num_overfit_transforms):
                a = random_se3(
                    1,
                    rot_var=self.rot_var,
                    trans_var=self.trans_var,
                    device=points_action.device,
                )
                b = random_se3(
                    1,
                    rot_var=self.rot_var,
                    trans_var=self.trans_var,
                    device=points_anchor.device,
                )
                self.T0_list.append(a)
                self.T1_list.append(b)
        return

    def get_existing_data_indices(self):
        import fnmatch

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

    def load_data(self, filename, action_class, anchor_class):
        point_data = np.load(filename, allow_pickle=True)
        points_raw_np = point_data["clouds"]
        classes_raw_np = point_data["classes"]
        if self.min_num_cameras < 4:
            camera_idxs = np.concatenate(
                [[0], np.cumsum((np.diff(classes_raw_np) == -2))]
            )
            if not np.all(np.isin(np.arange(4), np.unique(camera_idxs))):
                print(
                    "\033[93m"
                    + f"{filename} did not contain all classes in all cameras"
                    + "\033[0m"
                )
                return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

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

        points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
        points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

        symmetric_cls = torch.Tensor([])

        return points_action, points_anchor, symmetric_cls

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
            if (points_action.shape[1] < self.num_points) or (
                points_anchor.shape[1] < self.num_points
            ):
                bad_demo_id.append(i)

        return bad_demo_id

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
        filename = self.filenames[torch.randint(len(self.filenames), [1])]
        points_action, points_anchor, symmetric_cls = self.load_data(
            filename,
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )

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
