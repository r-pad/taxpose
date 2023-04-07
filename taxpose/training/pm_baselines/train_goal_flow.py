import math
import os
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch_geometric.loader as tgl
import typer

from taxpose.datasets.pm_placement import SEEN_CATS, UNSEEN_CATS, get_dataset_ids_all
from taxpose.training.pm_baselines.dataloader_goal_flow import create_gf_dataset
from taxpose.training.pm_baselines.flow_model import FlowNet as GoalInfFlowNet
from taxpose.training.pm_baselines.train_bc import WandBCallback

"""
This file contains training code for the naive goal inference module.
"""


def create_dataset_handles(dset, mode, nrep=1):
    envs_all = []
    split = math.floor(0.8 * len(dset))
    if mode == "train":
        for e in dset[:split]:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all
    else:
        for e in dset[split:]:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all


def create_dataset_handles_all(dset, nrep=1, random=False):
    envs_all = []
    if not random:
        for e in dset:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all
    else:
        a = np.arange(20 * 21 * 17)
        rand_idx = np.random.choice(a, size=nrep)
        for e in dset:
            for i in rand_idx:
                envs_all.append(f"{e}_{i}")
        return envs_all


def train(
    root: str = os.path.expanduser("~/datasets/partnet-mobility"),
    wandb: bool = True,
    epochs: int = 100,
    even_sampling: bool = False,
    randomize_camera: bool = False,
    n_workers: int = 60,
    n_proc_per_worker: int = 1,
):
    nrep = 200

    train_ids, val_ids, unseen_ids = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)
    train_envs = create_dataset_handles_all(train_ids, nrep, random=False)
    val_envs = create_dataset_handles_all(val_ids, 100, random=False)

    model: pl.LightningModule
    model = GoalInfFlowNet()

    n_repeat = 1

    train_dset = create_gf_dataset(
        root=root,
        obj_ids=train_envs,
        n_repeat=n_repeat,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=123456,
    )

    test_dset = create_gf_dataset(
        root=root,
        obj_ids=val_envs,
        n_repeat=1,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=654321,
    )

    train_loader = tgl.DataLoader(
        train_dset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=False, num_workers=0
    )

    logger: Optional[plog.WandbLogger]
    cbs: Optional[List[plc.Callback]]
    if wandb:
        logger = plog.WandbLogger(
            project=f"goal_inference_naive_rotation",
            config={
                "train_ids": train_envs,
                "test_ids": val_envs,
            },
        )
        model_dir = f"checkpoints/goal_inference_naive/{logger.experiment.name}"
        cbs = [
            WandBCallback(train_dset, test_dset),
            plc.ModelCheckpoint(dirpath=model_dir, every_n_epochs=1),
        ]

    else:
        logger = None
        cbs = None

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=cbs,
        log_every_n_steps=5,
        max_epochs=epochs,
    )

    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    if wandb and logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
