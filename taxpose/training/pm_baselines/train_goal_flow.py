import math
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
from taxpose.training.pm_baselines.dataloader_goal_flow import create_gf_dataset
from taxpose.training.pm_baselines.goal_flow import GoalInfFlowNet, maniskill_plot

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


class WandBCallback(plc.Callback):
    def __init__(self, train_dset, val_dset, eval_per_n_epoch: int = 1):
        self.train_dset = train_dset
        self.val_dset = val_dset
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
        # data.x = data.mask.reshape((-1, 1))
        obs_data = tgd.Batch.from_data_list([data]).to(pl_module.device)
        gdata = dset[randid][0]
        # gdata.x = gdata.mask.reshape((-1, 1))
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


def train(
    root: str = os.path.expanduser("~/datasets/partnet-mobility"),
    wandb: bool = True,
    epochs: int = 100,
    even_sampling: bool = False,
    randomize_camera: bool = False,
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
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_repeat=n_repeat,
        n_workers=60,
        n_proc_per_worker=1,
        seed=123456,
    )

    test_dset = create_gf_dataset(
        root=root,
        obj_ids=val_envs,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_repeat=1,
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
