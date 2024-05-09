import numpy as np
import pytest
import torch

import taxpose.utils.multilateration as multi


@pytest.mark.parametrize("dim", [2, 3])
def test_multilateration(dim):
    # Set the random seed.
    np.random.seed(0)

    P = np.random.uniform(low=-1.0, high=1.0, size=(101, dim, 1))
    P0, P = P[0], P[1:]
    R = np.linalg.norm((P - P0[None]).squeeze(), axis=-1)

    P = torch.from_numpy(np.asarray(P))
    R = torch.from_numpy(np.asarray(R))

    P0_est = multi.estimate_p(P, R).numpy()

    assert np.isclose(P0, P0_est).all()
