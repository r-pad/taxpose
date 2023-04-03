import pytest
import torch

from taxpose.train_pm_placement import classwise_mean, global_mean, t_err, theta_err

I = torch.eye(3, dtype=torch.float32)
Z_ROT_45 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
O = torch.zeros(3, dtype=torch.float32)
Z_T = torch.tensor([0, 0, 1], dtype=torch.float32)


@pytest.mark.parametrize("r1,r2", [(I, Z_ROT_45), (I[None], Z_ROT_45[None])])
def test_rotation_err(r1, r2):
    # Make sure it works both single or batched.
    err = theta_err(r1, r2).squeeze()
    assert torch.allclose(err, torch.tensor(90.0))


@pytest.mark.parametrize("t1,t2", [(O, Z_T), (O[None], Z_T[None])])
def test_t_err(t1, t2):
    # Make sure it works both single or batched.
    err = t_err(t1, t2).squeeze()
    assert torch.allclose(err, torch.tensor(1.0))


def test_aggregation():
    obj_results = {
        "10036": [
            {
                "R_err": 1.0,
                "T_err": 2.0,
            }
        ],
        "10068": [
            {
                "R_err": 3.0,
                "T_err": 4.0,
            }
        ],
    }

    cms = classwise_mean(obj_results)
    gm = global_mean(obj_results)

    assert cms["R_err"]["Fridge"] == 2.0
    assert cms["T_err"]["Fridge"] == 3.0
    assert gm["R_err"] == 2.0
    assert gm["T_err"] == 3.0
