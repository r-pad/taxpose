import torch
from torch import nn as nn
from torch.nn import functional as F

from third_party.dcp.model import get_graph_feature


class DGCNN_GC(nn.Module):
    """This is a goal-conditioned version of the DGCNN model from the DCP paper.

    See: https://github.com/WangYueFt/dcp/blob/master/model.py"""

    def __init__(self, emb_dims=512, input_dims=3, gc=False):
        super(DGCNN_GC, self).__init__()
        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        final_conv = 512 if not gc else 512 + 64
        self.conv5 = nn.Conv2d(final_conv, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
        if gc:
            self.cat_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU())

    def forward(self, x, cat=None):
        """Same as the original DGCNN model, but with an optional goal-conditioning."""
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        if cat is not None:
            x_cat = self.cat_mlp(cat.float())
            x_cat = x_cat.reshape(-1, 64, 1, 1).repeat(1, 1, x4.shape[2], x4.shape[3])
            x4 = torch.cat([x4, x_cat], dim=1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x
