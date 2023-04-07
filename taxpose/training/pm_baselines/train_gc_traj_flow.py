import os
from typing import List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer

from taxpose.datasets.pm_placement import SEEN_CATS, UNSEEN_CATS, get_dataset_ids_all
from taxpose.training.pm_baselines.dataloader_ff_interp_dagger import (
    create_gcdagger_dataset,
)
from taxpose.training.pm_baselines.train_bc import get_ids
from taxpose.training.pm_baselines.traj_flow import TrajFlowNet, maniskill_plot


class WandBCallback(plc.Callback):
    def __init__(
        self, train_dset, val_dset, unseen_dset=None, eval_per_n_epoch: int = 1
    ):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.unseen_dset = unseen_dset
        self.eval_per_n_epoch = eval_per_n_epoch

    @staticmethod
    def eval_log_random_sample(
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dset,
        prefix: Literal["train", "val", "unseen"],
    ):
        randid = np.random.randint(0, len(dset))
        data = dset[randid][1]
        data.x = data.mask.reshape((-1, 1))
        obs_data = tgd.Batch.from_data_list([data]).to(pl_module.device)
        gdata = dset[randid][0]
        gdata.x = gdata.mask.reshape((-1, 1))
        goal_data = tgd.Batch.from_data_list([gdata]).to(pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            f_pred = pl_module(goal_data, obs_data)

        assert trainer.logger is not None
        trainer.logger.experiment.log(
            {
                f"{prefix}/goal_cond_flow_plot": maniskill_plot(
                    obs_data.id,
                    goal_data.id,
                    obs_data.pos.cpu(),
                    goal_data.pos.cpu(),
                    goal_data.mask.cpu(),
                    obs_data.mask.cpu(),
                    f_pred.cpu(),
                    obs_data.flow.cpu(),
                ),
                "global_step": trainer.global_step,
            },
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.train_dset, "train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.val_dset, "val")
            if self.unseen_dset is not None:
                self.eval_log_random_sample(
                    trainer, pl_module, self.unseen_dset, "unseen"
                )


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
    model = TrajFlowNet()

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
        if model_type == "flownet":
            logger = plog.WandbLogger(
                project=f"goal_conditioned_traj_flow",
                config={
                    "train_ids": train_envs,
                    "test_ids": val_envs,
                    "unseen_ids": unseen_envs,
                },
            )
            model_dir = f"checkpoints/gc_traj_flow/{logger.experiment.name}"
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
