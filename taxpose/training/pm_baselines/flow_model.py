from dataclasses import dataclass
from typing import Dict

import plotly.graph_objects as go
import pytorch_lightning as pl
import rpad.pyg.nets.mlp as pnm
import rpad.pyg.nets.pointnet2 as pnp
import rpad.visualize_3d.plots as pvp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric.data as tgd
from plotly.subplots import make_subplots
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from taxpose.datasets.pm_placement import CATEGORIES


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


@dataclass
class FRNetCLIPortParams:
    in_dim: int = 1
    sa1: pnp.SAParams = pnp.SAParams(0.2, 0.2, pnm.MLPParams((64, 64)))
    sa1_outdim: int = 128

    sa2: pnp.SAParams = pnp.SAParams(0.25, 0.4, pnm.MLPParams((128, 128)))
    sa2_outdim: int = 256

    gsa: pnp.GlobalSAParams = pnp.GlobalSAParams(pnm.MLPParams((256, 512)))
    gsa_outdim: int = 1024

    # Parameters for the Feature Propagation modules.
    fp3: pnp.FPParams = pnp.FPParams(pnm.MLPParams((256, 256)), k=1)
    fp2: pnp.FPParams = pnp.FPParams(pnm.MLPParams((256, 128)))
    fp1: pnp.FPParams = pnp.FPParams(pnm.MLPParams((256, 128)))
    fp1_outdim: int = 128

    # Dimensions of the final 2 linear layers.
    lin1_dim: int = 128
    lin2_dim: int = 128

    # Final output dim
    final_outdim: int = 3


class FRNetCLIPort(nn.Module):
    def __init__(self, p: FRNetCLIPortParams):
        super().__init__()

        # The Set Aggregation modules.
        self.sa1 = pnp.SAModule(3 + p.in_dim, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa1)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules.
        self.fp3 = pnp.FPModule(p.gsa_outdim + p.sa2_outdim, p.sa2_outdim, p.fp3)
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + p.in_dim, p.fp1_outdim, p.fp1)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, 3)  # Flow output.

    def forward(self, data, flows):
        sa0_out = (data.x, data.pos.float(), data.batch)

        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        # CLIPort network uses tiling then hadamard product at each subsequent layer.
        x3 = torch.mul(x3, flows.tile(1, 8))
        sa3_out = x3, pos3, batch3

        fp3_x, fp3_pos, fp3_batch = self.fp3(*sa3_out, *sa2_out)
        temp = torch.cat(
            [
                flows[i].tile(int(fp3_x.shape[0] / flows.shape[0]), 2)
                for i in range(flows.shape[0])
            ]
        )
        if temp.shape[0] != fp3_x.shape[0]:
            temp = torch.cat([temp, flows[: fp3_x.shape[0] - temp.shape[0]].tile(1, 2)])
        fp3_x = torch.mul(fp3_x, temp)
        fp3_out = fp3_x, fp3_pos, fp3_batch
        fp2_x, fp2_pos, fp2_batch = self.fp2(*fp3_out, *sa1_out)
        temp = torch.cat(
            [
                flows[i].tile(int(fp2_x.shape[0] / flows.shape[0]), 1)
                for i in range(flows.shape[0])
            ]
        )
        if temp.shape[0] != fp2_x.shape[0]:
            temp = torch.cat([temp, flows[: fp2_x.shape[0] - temp.shape[0]].tile(1, 1)])
        fp2_x = torch.mul(fp2_x, temp)
        fp2_out = fp2_x, fp2_pos, fp2_batch
        x, _, _ = self.fp1(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


class FRNetCLIPortProjection(nn.Module):
    def __init__(self, p: FRNetCLIPortParams):
        super().__init__()

        # The Set Aggregation modules.
        self.sa1 = pnp.SAModule(3 + p.in_dim, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa1)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules.
        self.fp3 = pnp.FPModule(p.gsa_outdim + p.sa2_outdim, p.sa2_outdim, p.fp3)
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + p.in_dim, p.fp1_outdim, p.fp1)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, p.final_outdim)  # Flow output.
        self.proj1 = torch.nn.Linear(p.sa2_outdim // 2, p.sa2_outdim)
        self.proj2 = torch.nn.Linear(p.sa1_outdim, p.sa1_outdim)

    def forward(self, data, flows):
        sa0_out = (data.x, data.pos.float(), data.batch)

        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        # CLIPort network uses tiling then hadamard product at each subsequent layer.
        x3 = torch.mul(x3, flows.tile(1, 8))
        sa3_out = x3, pos3, batch3

        fp3_x, fp3_pos, fp3_batch = self.fp3(*sa3_out, *sa2_out)
        temp = torch.cat(
            [flows[i].tile((fp3_batch == i).sum(), 2) for i in range(flows.shape[0])]
        )
        fp3_x = torch.mul(fp3_x, temp)
        fp3_out = fp3_x, fp3_pos, fp3_batch
        fp2_x, fp2_pos, fp2_batch = self.fp2(*fp3_out, *sa1_out)
        temp = torch.cat(
            [flows[i].tile((fp2_batch == i).sum(), 1) for i in range(flows.shape[0])]
        )
        fp2_x = torch.mul(fp2_x, temp)
        fp2_out = fp2_x, fp2_pos, fp2_batch
        x, _, _ = self.fp1(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


@dataclass
class FlowNetParams:
    in_dim: int = 0
    flow_embed_dim: int = 128
    gfe_net: pnp.PN2EncoderParams = pnp.PN2EncoderParams()
    fr_net: FRNetCLIPortParams = FRNetCLIPortParams()
    inference: bool = False


class FlowNet(pl.LightningModule):
    def __init__(self, p: FlowNetParams = FlowNetParams()):
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
        n_nodes = torch.as_tensor([d.num_nodes for d in batch[0].to_data_list()]).to(self.device)  # type: ignore
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
