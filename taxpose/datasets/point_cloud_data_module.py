import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from taxpose.datasets.point_cloud_dataset import PointCloudDataset
from taxpose.datasets.point_cloud_dataset_test import TestPointCloudDataset


class MultiviewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root="/home/bokorn/src/ndf_robot/notebooks",
        test_dataset_root="/home/exx/Documents/ndf_robot/test_data/renders",
        dataset_index=10,
        action_class=0,
        anchor_class=1,
        dataset_size=1000,
        rotation_variance=np.pi / 180 * 5,
        translation_variance=0.1,
        batch_size=8,
        num_workers=8,
        cloud_type="final",
        symmetric_class=None,
        num_points=1024,
        overfit=False,
        gripper_lr_label=False,
        no_transform_applied=False,
        init_distribution_tranform_file="",
        num_demo=12,
        occlusion_class=0,
        cfg=None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.test_dataset_root = test_dataset_root
        if isinstance(dataset_index, list):
            self.dataset_index = dataset_index
        elif dataset_index == None:
            self.dataset_index = None
        self.dataset_index = dataset_index
        self.no_transform_applied = no_transform_applied

        self.action_class = action_class
        self.anchor_class = anchor_class
        self.dataset_size = dataset_size
        self.rotation_variance = rotation_variance
        self.translation_variance = translation_variance
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cloud_type = cloud_type
        self.symmetric_class = symmetric_class
        self.num_points = num_points
        self.overfit = overfit
        self.gripper_lr_label = gripper_lr_label
        self.index_list = []
        self.init_distribution_tranform_file = init_distribution_tranform_file
        self.num_demo = num_demo
        self.occlusion_class = occlusion_class

        self.cfg = cfg

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        """called only once and on 1 GPU"""

    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)

        if stage == "fit" or stage is None:
            # self.train_dataset = PointCloudDataset(
            #     cfg=PointClassDatasetConfig(
            #         demo_dset=NDFPointCloudDatasetConfig(
            #             dataset_root=self.dataset_root,
            #             dataset_indices=self.dataset_index,  # [self.dataset_index],
            #             num_demo=self.num_demo,
            #             min_num_points=self.num_points,
            #             cloud_type=self.cloud_type,
            #             action_class=self.action_class,
            #             anchor_class=self.anchor_class,
            #         ),
            #         num_points=self.num_points,
            #         rotation_variance=self.rotation_variance,
            #         translation_variance=self.translation_variance,
            #         dataset_size=self.dataset_size,
            #         synthetic_occlusion=self.synthetic_occlusion,
            #         ball_radius=self.ball_radius,
            #         plane_occlusion=self.plane_occlusion,
            #         ball_occlusion=self.ball_occlusion,
            #         plane_standoff=self.plane_standoff,
            #         occlusion_class=self.occlusion_class,
            #         overfit=self.overfit,
            #     )
            # )
            self.train_dataset = PointCloudDataset(self.cfg.train_dset)

        if stage == "val" or stage is None:
            # self.val_dataset = PointCloudDataset(
            #     cfg=PointClassDatasetConfig(
            #         demo_dset=NDFPointCloudDatasetConfig(
            #             dataset_root=self.test_dataset_root,
            #             dataset_indices=self.dataset_index,  # [self.dataset_index],
            #             num_demo=None,
            #             min_num_points=self.num_points,
            #             cloud_type=self.cloud_type,
            #             action_class=self.action_class,
            #             anchor_class=self.anchor_class,
            #         ),
            #         num_points=self.num_points,
            #         rotation_variance=self.rotation_variance,
            #         translation_variance=self.translation_variance,
            #         dataset_size=self.dataset_size,
            #         synthetic_occlusion=self.synthetic_occlusion,
            #         ball_radius=self.ball_radius,
            #         plane_occlusion=self.plane_occlusion,
            #         ball_occlusion=self.ball_occlusion,
            #         plane_standoff=self.plane_standoff,
            #         occlusion_class=self.occlusion_class,
            #         overfit=self.overfit,
            #     )
            # )
            self.val_dataset = PointCloudDataset(self.cfg.val_dset)
        if stage == "test":
            self.test_dataset = TestPointCloudDataset(
                dataset_root=self.test_dataset_root,
                dataset_indices=self.dataset_index,  # [self.dataset_index],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
                dataset_size=self.dataset_size,
                rotation_variance=self.rotation_variance,
                translation_variance=self.translation_variance,
                cloud_type=self.cloud_type,
                symmetric_class=self.symmetric_class,
                num_points=self.num_points,
                overfit=self.overfit,
                gripper_lr_label=self.gripper_lr_label,
                index_list=self.index_list,
                no_transform_applied=self.no_transform_applied,
                init_distribution_tranform_file=self.init_distribution_tranform_file,
            )

    def return_index_list_test(self):
        return self.test_dataset.return_index_list()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
