import os
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Transform3d
from torch.utils.data import Dataset

from taxpose.utils.se3 import pure_translation_se3, random_se3, rotation_se3


class TestPointCloudDataset(Dataset):
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
        index_list=[],
        no_transform_applied=False,
        init_distribution_tranform_file="",
    ):
        self.dataset_size = dataset_size
        self.num_points = num_points
        # Path('/home/bokorn/src/ndf_robot/notebooks')
        self.dataset_root = Path(dataset_root)
        self.cloud_type = cloud_type
        self.cloud_type_init = "init"
        self.rot_var = rotation_variance
        self.trans_var = translation_variance
        self.action_class = action_class
        self.anchor_class = anchor_class
        self.symmetric_class = symmetric_class  # None if no symmetric class exists
        self.angle_degree = angle_degree
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
        self.no_transform_applied = no_transform_applied

        print("dataset_root")
        print(dataset_root)
        self.init_distribution_tranform_file = init_distribution_tranform_file
        self.filenames = [
            self.dataset_root / f"{idx}_{self.cloud_type_init}_obj_points.npz"
            for idx in sorted(self.dataset_indices)
            if idx not in self.bad_demo_id
        ]

        if self.overfit:
            self.filenames = self.filenames[0:1]
        shapenet_id = np.load(self.filenames[0], allow_pickle=True)["shapenet_id"]
        print("shapenet_id:{}".format(shapenet_id))

        self.get_fixed_transforms()
        self.dataset_size = len(self.filenames)
        print("TestPointCloudDataset")
        print("data count:{}".format(len(self.filenames)))
        self.index_list = index_list

        print(self.filenames)

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
        elif self.init_distribution_tranform_file is not None:
            transform_init2place = np.load(
                self.init_distribution_tranform_file, allow_pickle=True
            )[
                "rack_relative_pose_stacked"
            ]  # 100,4,4

            for i in range(len(transform_init2place)):
                tmp = random_se3(
                    1,
                    rot_var=self.rot_var,
                    trans_var=self.trans_var,
                    device=points_action.device,
                )
                bm = (
                    torch.transpose(torch.from_numpy(transform_init2place[i]), -1, -2)
                    .unsqueeze(0)
                    .float()
                )

                bm[:, 3, :3] *= -1

                a = Transform3d(matrix=bm, device=points_action.device)
                self.T0_list.append(a.inverse())

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

        points_action_np = points_raw_np[classes_raw_np == action_class].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
        points_anchor_np = points_anchor_np - points_action_mean_np
        points_anchor_mean_np = points_anchor_np.mean(axis=0)

        points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
        points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

        symmetric_cls = torch.Tensor([])

        if (action_class == 2) and self.gripper_lr_label:
            action_center = torch.from_numpy(
                point_data["gripper_pose"][0] - points_action_mean_np
            ).view(1, 1, 3)
            action_vec = torch.from_numpy(point_data["gripper_pose"][2][:, 1]).view(
                1, 1, 3
            )

            action_centered = points_action - action_center
            anchor_mean = points_anchor.mean(dim=1, keepdim=True)
            anchor_vec = anchor_mean - action_center

            symmetric_cls = torch.sign(
                torch.matmul(action_centered, action_vec.transpose(-1, -2))
            ) == torch.sign(torch.matmul(anchor_vec, action_vec.transpose(-1, -2)))
            symmetric_cls = symmetric_cls.view(-1)

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

    def return_index_list(self):
        return self.index_list

    def __getitem__(self, index):
        self.index_list.append(index)

        filename = self.filenames[index]

        points_action, points_anchor, symmetric_cls = self.load_data(
            filename,
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )

        if self.overfit:
            transform_idx = torch.randint(self.num_overfit_transforms, (1,)).item()
            T0 = self.T0_list[transform_idx]
            T1 = self.T1_list[transform_idx]
        elif self.init_distribution_tranform_file is not None:
            transform_idx = torch.randint(len(self.T0_list), (1,)).item()
            T0 = self.T0_list[transform_idx]
        else:
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
            points_action, action_ids = sample_farthest_points(
                points_action, K=self.num_points, random_start_point=True
            )
            if len(symmetric_cls) > 0:
                symmetric_cls = symmetric_cls[action_ids.view(-1)]
        elif points_action.shape[1] < self.num_points:
            raise NotImplementedError(
                f"Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})"
            )

        if points_anchor.shape[1] > self.num_points:
            points_anchor, _ = sample_farthest_points(
                points_anchor, K=self.num_points, random_start_point=True
            )
        elif points_anchor.shape[1] < self.num_points:
            raise NotImplementedError(
                f"Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})"
            )

        if self.symmetric_class is not None:
            z_axis = torch.tensor([0, 0, 1])
            T_flip = rotation_se3(
                1,
                axis=z_axis,
                angle_degree=self.angle_degree,
                device=points_action.device,
            )
            if self.symmetric_class == self.action_class:
                T_trans_centered = pure_translation_se3(
                    1,
                    t=-points_action.squeeze(0).mean(axis=0),
                    device=points_action.device,
                )
                T_trans_uncentered = pure_translation_se3(
                    1,
                    t=points_action.squeeze(0).mean(axis=0),
                    device=points_action.device,
                )
                points_action_alternative = T_trans_uncentered.transform_points(
                    T_flip.transform_points(
                        T_trans_centered.transform_points(points_action)
                    )
                )
                points_anchor_alternative = points_anchor
                before_flip_original = points_action
                after_flip_original = points_action_alternative
            elif self.symmetric_class == self.anchor_class:
                T_trans_centered = pure_translation_se3(
                    1,
                    t=-points_anchor.squeeze(0).mean(axis=0),
                    device=points_action.device,
                )
                T_trans_uncentered = pure_translation_se3(
                    1,
                    t=points_anchor.squeeze(0).mean(axis=0),
                    device=points_action.device,
                )
                points_action_alternative = points_action
                points_anchor_alternative = T_trans_uncentered.transform_points(
                    T_flip.transform_points(
                        T_trans_centered.transform_points(points_anchor)
                    )
                )
                before_flip_original = points_anchor
                after_flip_original = points_anchor_alternative

        if self.init_distribution_tranform_file is not None:
            points_action_trans = T0.transform_points(points_action)
            points_anchor_trans = points_anchor
        else:
            points_action_trans = T0.transform_points(points_action)
            points_anchor_trans = T1.transform_points(points_anchor)
        if self.symmetric_class is not None:
            points_action_trans_alt = T0.transform_points(points_action_alternative)
            points_anchor_trans_alt = T1.transform_points(points_anchor_alternative)
            if self.symmetric_class == self.action_class:
                before_flip_transformed = points_action_trans
                after_flip_transformed = points_action_trans_alt
            elif self.symmetric_class == self.anchor_class:
                before_flip_transformed = points_anchor_trans
                after_flip_transformed = points_anchor_trans_alt

        if self.no_transform_applied or self.cloud_type == "init":
            data = {
                "points_action_trans": points_action.squeeze(0),
                "points_anchor_trans": points_anchor.squeeze(0),
            }
        elif self.init_distribution_tranform_file is not None:
            data = {
                "points_action_trans": points_action_trans.squeeze(0),
                "points_anchor_trans": points_anchor_trans.squeeze(0),
            }
        else:
            data = {
                "points_action": points_action.squeeze(0),
                "points_anchor": points_anchor.squeeze(0),
                "points_action_trans": points_action_trans.squeeze(0),
                "points_anchor_trans": points_anchor_trans.squeeze(0),
                "T0": T0.get_matrix().squeeze(0),
                "T1": T1.get_matrix().squeeze(0),
                "points_action": points_action.squeeze(0),
                "symmetric_cls": symmetric_cls,
            }
            if self.symmetric_class is not None:
                data["points_action_trans_alt"] = points_action_trans_alt.squeeze(0)
                data["points_anchor_trans_alt"] = points_anchor_trans_alt.squeeze(0)
                data["before_flip_original"] = before_flip_original.squeeze(0)
                data["after_flip_original"] = after_flip_original.squeeze(0)
                data["before_flip_transformed"] = before_flip_transformed.squeeze(0)
                data["after_flip_transformed"] = after_flip_transformed.squeeze(0)

        return data

    def __len__(self):
        return self.dataset_size
