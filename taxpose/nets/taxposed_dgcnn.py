import torch
from torch import nn as nn
from torch.nn import functional as F

from third_party.dcp.model import get_graph_feature


class DGCNN(nn.Module):
    """This is a modified version of the DGCNN model from the DCP paper
    for variable size inputs and conditioning in the later conv layers

    See: https://github.com/WangYueFt/dcp/blob/master/model.py"""

    def __init__(
        self,
        emb_dims=512,
        input_dims=3,
        num_heads=1,
        conditioning_size=0,
        last_relu=True,
    ):
        super(DGCNN, self).__init__()
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.conditioning_size = conditioning_size
        self.last_relu = last_relu

        self.conv1 = nn.Conv2d(2 * input_dims, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        if self.num_heads == 1:
            self.conv5 = nn.Conv2d(
                512 + self.conditioning_size, emb_dims, kernel_size=1, bias=False
            )
            self.bn5 = nn.BatchNorm2d(emb_dims)
        else:
            if self.conditioning_size > 0:
                raise NotImplementedError(
                    "Conditioning not implemented for multi-head DGCNN"
                )
            self.conv5s = nn.ModuleList(
                [
                    nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
                    for _ in range(self.num_heads)
                ]
            )
            self.bn5s = nn.ModuleList(
                [nn.BatchNorm2d(emb_dims) for _ in range(self.num_heads)]
            )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x, conditioning=None):
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

        if self.conditioning_size == 0:
            assert conditioning is None
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            assert conditioning is not None
            x = torch.cat((x1, x2, x3, x4, conditioning[:, :, :, None]), dim=1)

        if self.num_heads == 1:
            x = self.bn5(self.conv5(x)).view(batch_size, -1, num_points)
        else:
            x = [
                bn5(conv5(x)).view(batch_size, -1, num_points)
                for bn5, conv5 in zip(self.bn5s, self.conv5s)
            ]

        if self.last_relu:
            if self.num_heads == 1:
                x = F.relu(x)
            else:
                x = [F.relu(head) for head in x]
        return x


class DGCNNClassification(nn.Module):
    # Reference: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py#L88-L153

    def __init__(
        self, emb_dims=512, input_dims=3, num_heads=1, dropout=0.5, output_channels=40
    ):
        super(DGCNNClassification, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        if self.num_heads == 1:
            self.linear3 = nn.Linear(256, self.output_channels)
        else:
            self.linear3s = nn.ModuleList(
                [nn.Linear(256, self.output_channels) for _ in range(self.num_heads)]
            )

    def forward(self, x):
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

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x).squeeze()
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        if self.num_heads == 1:
            x = self.linear3(x)[:, :, None]
        else:
            x = [linear3(x)[:, :, None] for linear3 in self.linear3s]
        return x
