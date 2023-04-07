import os
from typing import List, Optional

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch_geometric.loader as tgl
import typer

from taxpose.datasets.pm_placement import SEEN_CATS, UNSEEN_CATS, get_dataset_ids_all
from taxpose.training.pm_baselines.bc import BCNet as DaggerNet
from taxpose.training.pm_baselines.dataloader_ff_interp_dagger import (
    create_gcdagger_dataset,
)
from taxpose.training.pm_baselines.train_bc import WandBCallback, get_ids


def train(
    root: str = os.path.expanduser("~/datasets/partnet-mobility"),
    freefloat_dset: str = "./data/free_floating_traj_interp_multigoals",
    wandb: bool = True,
    mask_flow: bool = False,
    epochs: int = 100,
    model_type: str = "flownet",
    even_sampling: bool = False,
    randomize_camera: bool = False,
    n_workers: int = 60,
    n_proc_per_worker: int = 1,
):
    # We're doing batch training so don't do too much.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"

    train_ids, val_ids, unseen_ids = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)
    train_envs = get_ids(freefloat_dset, train_ids)
    val_envs = get_ids(freefloat_dset, val_ids)
    unseen_envs = None

    model: pl.LightningModule
    model = DaggerNet()

    n_repeat = 1

    train_dset = create_gcdagger_dataset(
        root=root,
        freefloat_dset_path=freefloat_dset,
        obj_ids=train_envs,
        n_repeat=n_repeat,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=123456,
    )
    test_dset = create_gcdagger_dataset(
        root=root,
        freefloat_dset_path=freefloat_dset,
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
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=False, num_workers=0
    )

    logger: Optional[plog.WandbLogger]
    cbs: Optional[List[plc.Callback]]
    if wandb:
        if model_type == "flownet":
            logger = plog.WandbLogger(
                project=f"goal_conditioned_dagger",
                config={
                    "train_ids": train_envs,
                    "test_ids": val_envs,
                    "unseen_ids": unseen_envs,
                },
            )
            model_dir = f"checkpoints/gc_dagger/{logger.experiment.name}"
            cbs = [
                WandBCallback(train_dset, test_dset),
                plc.ModelCheckpoint(dirpath=model_dir, every_n_epochs=1),
            ]
        else:
            raise ValueError("bad modeltype")

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

    val_loaders = test_loader
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)

    if wandb and logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
