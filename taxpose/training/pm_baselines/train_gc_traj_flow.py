import os
from typing import List, Literal, Optional

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as pvp
import torch
import torch.optim as opt
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import typer
from plotly.subplots import make_subplots
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from taxpose.datasets.pm_placement import SEEN_CATS, UNSEEN_CATS, get_dataset_ids_all
from taxpose.training.pm_baselines.dataloader_ff_interp_dagger import GCDaggerDataset
from taxpose.training.pm_baselines.naive_nets import FRNetCLIPort, GoalInfFlowNetParams


def artflownet_loss(
    f_pred: torch.Tensor,
    f_target: torch.Tensor,
    mask: torch.Tensor,
    n_nodes: torch.Tensor,
    use_mask=False,
) -> torch.Tensor:
    """Maniskill loss.
    Args:
        f_pred (torch.Tensor): Predicted flow.
        f_target (torch.Tensor): Target flow.
        mask (torch.Tensor): only mask
        n_nodes (torch.Tensor): A list describing the number of nodes
        use_mask: Whether or not to compute loss over all points, or just some.
    Returns:
        Loss
    """
    weights = (1 / n_nodes).repeat_interleave(n_nodes)

    if use_mask:
        f_pred = f_pred[mask]
        f_target = f_target[mask]
        weights = weights[mask]

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=1)
    # weight each PC equally in the sum.
    l_se = (raw_se * weights).sum()

    # Full loss.
    loss: torch.Tensor = l_se / len(n_nodes)

    return loss


class TrajFlowNet(pl.LightningModule):
    def __init__(self, p: GoalInfFlowNetParams = GoalInfFlowNetParams()):
        super().__init__()

        self.gfe_net = pnp.PN2Encoder(in_dim=1, out_dim=p.flow_embed_dim, p=p.gfe_net)
        self.fr_net = FRNetCLIPort(p.fr_net)

    @staticmethod
    def flow_metrics(pred_flow, gt_flow):
        with torch.no_grad():
            # RMSE
            rmse = (pred_flow - gt_flow).norm(p=2, dim=1).mean()

            # Cosine similarity, normalized.
            nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=1) != 0.0)
            gt_flow_nz = gt_flow[nonzero_gt_flowixs]
            pred_flow_nz = pred_flow[nonzero_gt_flowixs]
            cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=1).mean()

            # Magnitude
            mag_error = (
                (pred_flow.norm(p=2, dim=1) - gt_flow.norm(p=2, dim=1)).abs().mean()
            )
        return rmse, cos_dist, mag_error

    def forward(self, src_data, dst_data):  # type: ignore
        flow_embedding = self.gfe_net(src_data)

        pred_flow = self.fr_net(dst_data, flow_embedding)

        return pred_flow

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

        data = Data(pos=xyz, mask=mask, x=mask.reshape(-1, 1))
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        data_goal = Data(pos=xyz_goal, mask=mask_goal, x=mask_goal.reshape(-1, 1))
        batch_goal = Batch.from_data_list([data_goal])
        batch_goal = batch_goal.to(self.device)
        self.eval()
        with torch.no_grad():
            flow: torch.Tensor = self.forward(batch_goal, batch)
        return flow

    def _step(self, batch: tgd.Batch, mode):
        n_nodes = torch.as_tensor([d.num_nodes for d in batch[0].to_data_list()]).to(
            self.device
        )
        # dst_data: object of interest
        # src_data: transferred object
        src_data, dst_data = batch
        src_data.x = src_data.mask.reshape((-1, 1))
        dst_data.x = dst_data.mask.reshape((-1, 1))
        src_data, dst_data = src_data.to(self.device), dst_data.to(self.device)
        f_pred = self(src_data, dst_data)
        f_ix = dst_data.mask.bool()
        f_pred[~f_ix] = 0.0
        f_target = dst_data.flow
        loss = artflownet_loss(f_pred, f_target, dst_data.mask, n_nodes)
        rmse, cos_dist, mag_error = self.flow_metrics(f_pred[f_ix], f_target[f_ix])
        if not torch.isnan(rmse):
            self.log_dict(
                {
                    f"{mode}/loss": loss,
                    f"{mode}/rmse": rmse,
                    f"{mode}/cosine_similarity": cos_dist,
                    f"{mode}/mag_error": mag_error,
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


def maniskill_plot(
    obj_id, goal_id, pos, goal_pos, goal_mask, mask, f_pred, f_target
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "table"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Goal Condition",
            "N/A",
            "target flow",
            "pred flow",
        ),
    )
    labelmap = {0: "Microwave", 1: "Block"}
    labels = torch.zeros(len(goal_pos)).int()
    labels[goal_mask == 1.0] = 1
    fig.add_traces(pvp._segmentation_traces(goal_pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=pvp._3d_scene(goal_pos),
        legend=dict(x=1.0, y=0.75),
    )

    # Connectedness table.
    fig.append_trace(
        go.Table(
            header=dict(values=["IGNORE", "IGNORE"]),
            cells=dict(values=[[1.0], [1.0]]),
        ),
        row=1,
        col=2,
    )

    f_pred[~(mask == 1.0)] = 0

    # GT flow.
    labelmap = {0: "Microwave", 1: "Block"}
    labels = torch.zeros(len(pos)).int()
    labels[mask == 1.0] = 1
    fig.add_traces(
        pvp._segmentation_traces(pos + f_target, labels, labelmap, scene="scene2"),
    )

    fig.update_layout(scene2=pvp._3d_scene(pos + f_target))

    # Predicted flow.
    fig.add_traces(
        pvp._segmentation_traces(pos + f_pred, labels, labelmap, scene="scene3"),
    )
    fig.update_layout(scene3=pvp._3d_scene(pos + f_pred))

    fig.update_layout(title=f"Goal {goal_id}")

    return fig


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


def get_ids(dset, ids, nrep=1):
    all_trajs = os.listdir(dset)[:]
    envs_all = set()
    for oid in ids:
        for traj_name in all_trajs:
            if oid in traj_name and traj_name.endswith("npy"):
                traj_len = len(np.load(f"{dset}/{traj_name}"))
                for idx in range(traj_len * 10):
                    envs_all.add(f"{traj_name[:-4]}_{idx}")
    return list(envs_all)


def train(
    root: str = os.path.expanduser("~/partnet-mobility"),
    wandb: bool = True,
    mask_flow: bool = False,
    epochs: int = 60,
    model_type: str = "flownet",
    process: bool = True,
    even_sampling: bool = False,
    randomize_camera: bool = False,
):
    # We're doing batch training so don't do too much.
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["NUMEXPR_MAX_THREADS"] = "10"

    freefloat_dset = os.path.expanduser(
        f"~/discriminative_embeddings/part_embedding/goal_inference/baselines/free_floating_traj_interp_multigoals"
    )
    train_ids, val_ids, unseen_ids = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)
    train_envs = get_ids(freefloat_dset, train_ids)
    val_envs = get_ids(freefloat_dset, val_ids)[::500]
    unseen_envs = None

    model: pl.LightningModule
    model = TrajFlowNet()

    nrepeat = 1

    train_dset = GCDaggerDataset(
        root=root,
        obj_ids=train_envs,
        nrepeat=nrepeat,
        process=process,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
    )
    test_dset = GCDaggerDataset(
        root=root,
        obj_ids=val_envs,
        nrepeat=1,
        process=process,
        even_sampling=even_sampling,
        randomize_camera=randomize_camera,
    )

    train_loader = tgl.DataLoader(
        train_dset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        # worker_init_fn=_init_fn,
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=False, num_workers=0
    )

    unseen_dset = None
    unseen_loader = None
    if unseen_envs is not None:
        unseen_dset = GCDaggerDataset(
            root=root,
            obj_ids=unseen_envs,
            nrepeat=1,
            process=process,
            even_sampling=even_sampling,
            randomize_camera=randomize_camera,
        )
        unseen_loader = tgl.DataLoader(
            unseen_dset,
            batch_size=min(len(unseen_dset), 64),
            shuffle=False,
            num_workers=0,
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
                WandBCallback(train_dset, test_dset, unseen_dset),
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

    val_loaders = (
        [test_loader, unseen_loader] if unseen_loader is not None else test_loader
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)

    if wandb and logger is not None:
        logger.experiment.finish()


def load_env_names(path: str):
    with open(path, "r") as f:
        lines = f.read()
    return lines.split("\n")[:-1]


if __name__ == "__main__":
    typer.run(train)
