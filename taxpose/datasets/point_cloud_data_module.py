import joblib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from taxpose.datasets.point_cloud_dataset import PointCloudDataset


def parallel_iterate(dataset, n_jobs):
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(dataset.__getitem__)(ix) for ix in tqdm(range(len(dataset)))
    )

    # Iterate once over the dataset to make sure it's been processed.
    # _ = [dataset[ix] for ix in tqdm(range(len(dataset)))]


class MultiviewDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=8, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        """called only once and on 1 GPU"""

    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)

        if stage == "train" or stage is None:
            self.train_dataset = PointCloudDataset(self.cfg.train_dset)
            # Iterate once over the dataset to make sure it's been processed.
            parallel_iterate(self.train_dataset.dataset, self.num_workers)

        if stage == "val" or stage is None:
            self.val_dataset = PointCloudDataset(self.cfg.val_dset)
            # Iterate once over the dataset to make sure it's been processed.
            parallel_iterate(self.val_dataset.dataset, self.num_workers)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
