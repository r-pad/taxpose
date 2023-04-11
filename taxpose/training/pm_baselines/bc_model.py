from typing import Dict

import plotly.graph_objs as go
import pytorch_lightning as pl
import rpad.pyg.nets.pointnet2 as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric.data as tgd
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from third_party.dcp.model import DGCNN


class BCNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.dgcnn_0 = DGCNN(emb_dims=512)
        self.dgcnn_1 = DGCNN(emb_dims=512)
        self.lin = nn.Linear(512, 7)
        self.bc_loss = nn.MSELoss()

    @staticmethod
    def norm_scale(pos):
        mean = pos.mean(dim=1).unsqueeze(1)
        pos = pos - mean
        scale = pos.abs().max(dim=2)[0].max(dim=1)[0]
        pos = pos / (scale.view(-1, 1, 1) + 1e-8)
        return pos

    def forward(self, src_data, dst_data):
        dst_pos = self.norm_scale(dst_data.pos.view(-1, 2000, 3)).transpose(-1, -2)
        src_pos = self.norm_scale(src_data.pos.view(-1, 2000, 3)).transpose(-1, -2)
        out_0 = self.dgcnn_0(dst_pos.cuda())
        out_1 = self.dgcnn_1(src_pos.cuda())
        out = torch.multiply(out_0, out_1)
        out = F.relu(out)
        out = self.lin(out.transpose(1, 2))
        out = out.mean(axis=1)
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

    @staticmethod
    def make_plots(
        preds, obs_batch: tgd.Batch, goal_batch: tgd.Batch
    ) -> Dict[str, go.Figure]:
        # No plots, this is behavior cloning.
        return {}
