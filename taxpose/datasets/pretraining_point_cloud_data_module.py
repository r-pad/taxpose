import pytorch_lightning as pl
from torch.utils.data import DataLoader

from taxpose.datasets.ndf_dataset import JointOccTrainDataset
from taxpose.datasets.pretraining_point_cloud_dataset import (
    PretrainingPointCloudDataset,
)


class PretrainingMultiviewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cloud_class=0,
        batch_size=8,
        num_workers=8,
        cloud_type="final",
        dataset_index=None,
        dataset_root=None,
        obj_class="mug",
        pretraining_data_path=None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cloud_class = cloud_class
        self.cloud_type = cloud_type
        self.dataset_indices = dataset_index
        self.dataset_root = dataset_root
        self.pretraining_data_path = pretraining_data_path
        self.obj_class = obj_class

        # 0 for mug, 1 for rack, 2 for gripper
        if self.cloud_class == 0:
            self.obj_class = obj_class
        else:
            self.obj_class = "non_mug"

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        """called only once and on 1 GPU"""

    def update_dataset(self):
        if self.obj_class != "non_mug":
            self.train_dataset = JointOccTrainDataset(
                ndf_data_path=self.pretraining_data_path,
                obj_class=[self.obj_class],
                phase="train",
            )
        else:
            self.train_dataset = PretrainingPointCloudDataset(
                dataset_root=self.dataset_root,
                dataset_indices=self.dataset_indices,
                cloud_type=self.cloud_type,
                action_class=self.cloud_class,
            )

    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)

        if stage == "fit" or stage is None:
            print("TRAIN Dataset")
            if self.obj_class != "non_mug":
                self.train_dataset = JointOccTrainDataset(
                    ndf_data_path=self.pretraining_data_path,
                    obj_class=[self.obj_class],
                    phase="train",
                )
            else:
                self.train_dataset = PretrainingPointCloudDataset(
                    dataset_root=self.dataset_root,
                    dataset_indices=self.dataset_indices,
                    cloud_type=self.cloud_type,
                    action_class=self.cloud_class,
                )

        if stage == "val" or stage is None:
            print("VAL Dataset")
            if self.obj_class != "non_mug":
                self.val_dataset = JointOccTrainDataset(
                    ndf_data_path=self.pretraining_data_path,
                    obj_class=[self.obj_class],
                    phase="val",
                )
            else:
                self.val_dataset = PretrainingPointCloudDataset(
                    dataset_root=self.dataset_root,
                    dataset_indices=self.dataset_indices,
                    cloud_type=self.cloud_type,
                    action_class=self.cloud_class,
                )
        if stage == "test":
            if self.obj_class != "non_mug":
                self.test_dataset = JointOccTrainDataset(
                    ndf_data_path=self.pretraining_data_path, obj_class=[self.obj_class]
                )
            else:
                self.test_dataset = PretrainingPointCloudDataset(
                    dataset_root=self.dataset_root,
                    dataset_indices=self.dataset_indices,
                    cloud_type=self.cloud_type,
                    action_class=self.cloud_class,
                )

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
