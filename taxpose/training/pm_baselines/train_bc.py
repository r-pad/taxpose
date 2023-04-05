import os
from typing import List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import rpad.pyg.nets.pointnet2 as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from taxpose.datasets.pm_placement import get_dataset_ids_all
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


class BCNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.gfe_net_0 = pnp.PN2Encoder(in_dim=1, out_dim=128, p=pnp.PN2EncoderParams())
        self.gfe_net_1 = pnp.PN2Encoder(in_dim=1, out_dim=128, p=pnp.PN2EncoderParams())
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 3)
        self.bc_loss = nn.MSELoss()

    def forward(self, src_data, dst_data):  # type: ignore
        src_embed = self.gfe_net_0(src_data)
        dst_embed = self.gfe_net_1(dst_data)
        embed = torch.concat([src_embed, dst_embed], axis=1)
        out = F.relu(self.lin1(embed))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)

        return out

    def predict(
        self,
        xyz: torch.Tensor,
        mask: torch.Tensor,
        xyz_goal: torch.Tensor,
        mask_goal: torch.Tensor,
    ) -> torch.Tensor:
        assert len(xyz) == len(mask)
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 1

        data = Data(pos=xyz, mask=mask, x=mask.reshape((-1, 1)))
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        data_goal = Data(pos=xyz_goal, mask=mask_goal, x=mask_goal.reshape((-1, 1)))
        batch_goal = Batch.from_data_list([data_goal])
        batch_goal = batch_goal.to(self.device)
        self.eval()
        with torch.no_grad():
            flow: torch.Tensor = self.forward(batch_goal, batch)
        return flow

    def _step(self, batch: tgd.Batch, mode):
        # dst_data: object of interest
        # src_data: transferred object
        src_data, dst_data = batch
        src_data.x = src_data.mask.reshape((-1, 1))
        dst_data.x = dst_data.mask.reshape((-1, 1))
        src_data, dst_data = src_data.to(self.device), dst_data.to(self.device)
        f_pred = self(src_data, dst_data).reshape((-1,))
        f_target = dst_data.action
        loss = self.bc_loss(f_pred, f_target)
        self.log_dict(
            {
                f"{mode}/loss": loss,
            },
            add_dataloader_idx=False,
        )

        return loss

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        return self._step(batch, "train")

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()
        name = "val" if dataloader_idx == 0 else "unseen"
        return self._step(batch, name)

    def configure_optimizers(self):
        return opt.Adam(params=self.parameters(), lr=0.0001)


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
        n_workers=60,
        n_proc_per_worker=1,
        seed=123456,
    )
    print("created train dataset")

    test_dset = create_gcbc_dataset(
        root=root,
        freefloat_dset_path=freefloat_dset,
        obj_ids=val_envs,
        n_repeat=1,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
        seed=654321,
    )
    print("created test dataset")

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

    val_loaders = test_loader
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)

    if wandb and logger is not None:
        logger.experiment.finish()


def load_env_names(path: str):
    with open(path, "r") as f:
        lines = f.read()
    return lines.split("\n")[:-1]


if __name__ == "__main__":
    typer.run(train)
