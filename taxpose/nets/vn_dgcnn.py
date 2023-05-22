# Adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_dgcnn_partseg.py
# Only changes:
# - Change the paths to relative imports.
# - Make the label optional (i.e. goal-conditioned).
# - Add this comment.
from dataclasses import dataclass

import torch.nn as nn
import torch.utils.data

from third_party.vnn.utils.vn_dgcnn_util import get_graph_feature
from third_party.vnn.vn_layers import *


@dataclass
class VNArgs:
    n_knn: int
    pooling: str


class VN_DGCNN(nn.Module):
    def __init__(
        self, args, num_part=50, normal_channel=False, gc=True, num_gc_classes=16
    ):
        super(VN_DGCNN, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.gc = gc
        self.num_gc_classes = num_gc_classes

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        if args.pooling == "max":
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == "mean":
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearLeakyReLU(
            64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True
        )
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        f_dim = 2299 if self.gc else 2235
        self.conv8 = nn.Sequential(
            nn.Conv1d(f_dim, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2),
        )
        if self.gc:
            self.conv7 = nn.Sequential(
                nn.Conv1d(num_gc_classes, 64, kernel_size=1, bias=False),
                self.bn7,
                nn.LeakyReLU(negative_slope=0.2),
            )

        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            self.bn9,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            self.bn10,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)

    def forward(self, x, l=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum("bijm,bjkm->bikm", x123, z0).view(
            batch_size, -1, num_points
        )
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        # Modified from original.
        if self.gc:
            l = l.view(batch_size, -1, 1)
            l = self.conv7(l)

            x = torch.cat((x, l), dim=1)

        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        trans_feat = None
        return x.transpose(1, 2), trans_feat
