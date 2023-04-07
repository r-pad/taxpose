import abc
import os
from typing import List, Literal, Optional, Protocol

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer

from taxpose.datasets.pm_placement import get_dataset_ids_all
from taxpose.training.pm_baselines.bc_model import BCNet
from taxpose.training.pm_baselines.dataloader_ff_interp_bc import create_gcbc_dataset

SEEN_CATS = [
    "microwave",
    "dishwasher",
    "chair",
    "oven",
    "fridge",
    "safe",
    "table",
]
UNSEEN_CATS = ["drawer", "washingmachine"]


class CanMakePlots(Protocol):
    @staticmethod
    @abc.abstractmethod
    def make_plots(preds, obs_batch: tgd.Batch, goal_batch: tgd.Batch):
        pass


class LightningModuleWithPlots(pl.LightningModule, CanMakePlots):
    pass


class WandBCallback(plc.Callback):
    def __init__(self, train_dset, val_dset, eval_per_n_epoch: int = 1):
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.eval_per_n_epoch = eval_per_n_epoch

    @staticmethod
    def eval_log_random_sample(
        trainer: pl.Trainer,
        pl_module: LightningModuleWithPlots,
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

        plots = pl_module.make_plots(f_pred, obs_data, goal_data)

        logger = trainer.logger
        assert logger is not None and isinstance(logger, plog.WandbLogger)
        logger.experiment.log(
            {
                **{f"{prefix}/{plot_name}": plot for plot_name, plot in plots.items()},
                "global_step": trainer.global_step,
            },
            step=trainer.global_step,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.train_dset, "train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):  # type: ignore
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, self.val_dset, "val")


def get_ids(dset, ids, nrep=1):
    all_trajs = os.listdir(dset)[:]
    envs_all = set()
    for oid in ids:
        for traj_name in all_trajs:
            if oid in traj_name and traj_name.endswith("npy"):
                traj_len = len(np.load(f"{dset}/{traj_name}")) - 1
                for idx in range(traj_len):
                    envs_all.add(f"{traj_name[:-4]}_{idx}")
    return list(envs_all)


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
    model = BCNet()

    n_repeat = 1

    train_dset = create_gcbc_dataset(
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
    test_dset = create_gcbc_dataset(
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
        batch_size=8,
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
                project=f"taxpose_goal_conditioned_bc",
                config={
                    "train_ids": train_envs,
                    "test_ids": val_envs,
                    "unseen_ids": unseen_envs,
                },
            )
            model_dir = f"checkpoints/gc_bc/{logger.experiment.name}"
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

    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    if wandb and logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
