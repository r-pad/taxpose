from dataclasses import dataclass
from typing import Optional, cast

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from taxpose.datasets.base import (
    PretrainingPointCloudDataset,
    PretrainingPointCloudDatasetConfig,
)
from taxpose.datasets.ndf_pretraining import (
    NDFPretrainingPointCloudDataset,
    NDFPretrainingPointCloudDatasetConfig,
)
from taxpose.datasets.shapenet_pretraining import (
    ShapeNetPretrainingPointCloudDataset,
    ShapeNetPretrainingPointCloudDatasetConfig,
)


def make_dataset(cfg: PretrainingPointCloudDatasetConfig):
    dataset_type = cfg.dataset_type
    if dataset_type == "ndf_pretraining":
        ndf_cfg = cast(NDFPretrainingPointCloudDatasetConfig, cfg)
        return NDFPretrainingPointCloudDataset(ndf_cfg)
    elif dataset_type == "shapenet_pretraining":
        shapenet_cfg = cast(ShapeNetPretrainingPointCloudDatasetConfig, cfg)
        return ShapeNetPretrainingPointCloudDataset(shapenet_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


@dataclass
class PretrainingMultiviewDMConfig:
    train_dset: PretrainingPointCloudDatasetConfig
    val_dset: PretrainingPointCloudDatasetConfig


class PretrainingMultiviewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: PretrainingMultiviewDMConfig,
        batch_size=8,
        num_workers=8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg = cfg
        self.train_dataset: Optional[PretrainingPointCloudDataset] = None
        self.val_dataset: Optional[PretrainingPointCloudDataset] = None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = make_dataset(self.cfg.train_dset)
            self.val_dataset = make_dataset(self.cfg.val_dset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
