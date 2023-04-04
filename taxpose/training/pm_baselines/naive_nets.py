from dataclasses import dataclass

import rpad.pyg.nets.mlp as pnm
import rpad.pyg.nets.pointnet2 as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file contains implementation of the naive goal inference module.
"""


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
class GoalInfFlowNetParams:
    in_dim: int = 0
    flow_embed_dim: int = 128
    gfe_net: pnp.PN2EncoderParams = pnp.PN2EncoderParams()
    fr_net: FRNetCLIPortParams = FRNetCLIPortParams()
    inference: bool = False
