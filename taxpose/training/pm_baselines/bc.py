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

    @staticmethod
    def make_plots(preds, batch: tgd.Batch) -> Dict[str, go.Figure]:
        # No plots, this is behavior cloning.
        return {}
