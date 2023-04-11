import abc
import os
from typing import Literal, Protocol

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer

from taxpose.datasets.pm_placement import get_dataset_ids_all
from taxpose.training.pm_baselines.bc_dataset import (
    create_gcbc_dataset,
    create_gcdagger_dataset,
)
from taxpose.training.pm_baselines.bc_model import BCNet
from taxpose.training.pm_baselines.flow_dataset import create_gf_dataset
from taxpose.training.pm_baselines.flow_model import FlowNet

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


def create_dataset(
    model_type: str,
    root,
    freefloat_dset_path,
    obj_ids,
    n_repeat,
    even_sampling,
    randomize_camera,
    n_workers,
    n_proc_per_worker,
    seed,
    n_points=1200,
):
    if model_type == "bc":
        return create_gcbc_dataset(
            root=root,
            freefloat_dset_path=freefloat_dset_path,
            obj_ids=obj_ids,
            even_sampling=even_sampling,
            randomize_camera=randomize_camera,
            n_points=n_points,
            n_repeat=n_repeat,
            n_workers=n_workers,
            n_proc_per_worker=n_proc_per_worker,
            seed=seed,
        )
    elif model_type == "dagger" or model_type == "traj_flow":
        return create_gcdagger_dataset(
            root=root,
            freefloat_dset_path=freefloat_dset_path,
            obj_ids=obj_ids,
            even_sampling=even_sampling,
            randomize_camera=randomize_camera,
            n_points=n_points,
            n_repeat=n_repeat,
            n_workers=n_workers,
            n_proc_per_worker=n_proc_per_worker,
            seed=seed,
        )
    elif model_type == "goal_flow":
        return create_gf_dataset(
            root=root,
            obj_ids=obj_ids,
            even_sampling=even_sampling,
            randomize_camera=randomize_camera,
            n_points=n_points,
            n_repeat=n_repeat,
            n_workers=n_workers,
            n_proc_per_worker=n_proc_per_worker,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")


def train(
    root: str = os.path.expanduser("~/datasets/partnet-mobility"),
    freefloat_dset: str = "./data/free_floating_traj_interp_multigoals_rot",
    epochs: int = 100,
    model_type: str = "bc",
    even_sampling: bool = False,
    randomize_camera: bool = False,
    n_workers: int = 60,
    n_proc_per_worker: int = 1,
    batch_size: int = 32,
):
    assert model_type in ["bc", "dagger", "goal_flow", "traj_flow"]

    # We're doing batch training so don't do too much.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"

    train_ids, val_ids, unseen_ids = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)

    if model_type in {"bc", "dagger", "traj_flow"}:
        train_envs = get_ids(freefloat_dset, train_ids)
        val_envs = get_ids(freefloat_dset, val_ids)
    elif model_type == "goal_flow":
        nrep = 200
        train_envs = create_dataset_handles_all(train_ids, nrep, random=False)
        val_envs = create_dataset_handles_all(val_ids, 100, random=False)

    model: pl.LightningModule
    if model_type == "bc" or model_type == "dagger":
        model = BCNet()
    elif model_type == "goal_flow" or model_type == "traj_flow":
        model = FlowNet()
    else:
        raise ValueError(f"Unknown model type {model_type}")

    n_repeat = 1

    train_dset = create_dataset(
        model_type,
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
    print("Created train dataset")

    test_dset = create_dataset(
        model_type,
        root=root,
        freefloat_dset_path=freefloat_dset,
        obj_ids=val_envs,
        n_repeat=n_repeat,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=654321,
    )
    print("Created test dataset")

    train_loader = tgl.DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=False, num_workers=0
    )

    logger = plog.WandbLogger(
        project="taxpose_pm_baselines",
        config={
            "model_type": model_type,
            "train_ids": train_envs,
            "test_ids": val_envs,
        },
    )
    model_dir = f"checkpoints/{model_type}/{logger.experiment.name}"
    cbs = [
        WandBCallback(train_dset, test_dset),
        plc.ModelCheckpoint(dirpath=model_dir, every_n_epochs=1),
    ]

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=cbs,
        log_every_n_steps=5,
        max_epochs=epochs,
    )

    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    logger.experiment.finish()


if __name__ == "__main__":
    typer.run(train)
