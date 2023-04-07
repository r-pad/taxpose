from typing import Dict

import plotly.graph_objects as go
import pytorch_lightning as pl
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as pvp
import torch
import torch.optim as opt
import torch_geometric.data as tgd
from plotly.subplots import make_subplots
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from taxpose.datasets.pm_placement import CATEGORIES
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


def flow_plot(
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

    fig.update_layout(
        title=f"Goal {goal_id} Category {CATEGORIES[goal_id[0].split('_')[0]]}, Object {obj_id} Category {CATEGORIES[obj_id[0].split('_')[0]]}"
    )
    return fig


class FlowNet(pl.LightningModule):
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

    @staticmethod
    def make_plots(
        preds, obs_batch: tgd.Batch, goal_batch: tgd.Batch
    ) -> Dict[str, go.Figure]:
        # No plots, this is behavior cloning.
        return {
            "flow": flow_plot(
                obs_batch.id,
                goal_batch.id,
                obs_batch.pos.cpu(),
                goal_batch.pos.cpu(),
                goal_batch.mask.cpu(),
                obs_batch.mask.cpu(),
                preds.cpu(),
                obs_batch.flow.cpu(),
            )
        }
