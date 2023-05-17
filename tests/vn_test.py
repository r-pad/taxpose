from dataclasses import dataclass

import pytest
import torch
from scipy.spatial.transform import Rotation

from third_party.vnn.vn_dgcnn_partseg import get_model as VN_DGCNN


@dataclass
class Args:
    n_knn: int = 40
    pooling: str = "mean"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vn_dgcnn_invariance(seed):
    torch.manual_seed(seed)
    # Create a random point cloud.
    B = 1
    D = 3
    N = 100
    pos = torch.rand(B, D, N) * 2 - 1

    # Create a random rotation matrix.
    R = torch.as_tensor(Rotation.random().as_matrix(), dtype=torch.float32)

    # Apply the rotation to the point cloud.
    pos_rot = torch.matmul(R, pos)

    model = VN_DGCNN(Args, num_part=50, normal_channel=False, gc=False).cuda()
    model.eval()

    preds1, _ = model(pos.cuda())
    preds2, _ = model(pos_rot.cuda())

    assert torch.isclose(preds1, preds2, atol=1e-5).all()
