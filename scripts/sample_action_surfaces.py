import os
from pathlib import Path

import numpy as np
import pybullet as p
import typer
from rpad.partnet_mobility_utils.render.pybullet import Camera

from taxpose.datasets.pm_placement import ACTION_CLOUD_DIR
from taxpose.datasets.pm_utils import ACTION_OBJS


def sample_surface(urdf, d=0.2, scale=1) -> np.ndarray:
    pcs = []

    client_id = p.connect(p.DIRECT)
    p.loadURDF(
        urdf,
        physicsClientId=client_id,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        globalScaling=scale,
    )
    camera = Camera([1, 0, 0], target=[0, 0, 0])

    # six different sides. Not exactly above bc of my stupid lookAt code.
    for pos in [
        [0.01, 0, d],
        [-0.01, 0, -d],
        [d, 0, 0],
        [-d, 0, 0],
        [0, d, 0],
        [0, -d, 0],
    ]:
        camera.set_camera_position(pos)
        _, _, _, _, P_world, _, _ = camera.render(client_id, has_plane=False)
        pcs.append(P_world)

    return np.concatenate(pcs, axis=0)  # type: ignore


def main(out_dir: str = str(ACTION_CLOUD_DIR)):
    os.makedirs(ACTION_CLOUD_DIR, exist_ok=True)
    for name, obj in ACTION_OBJS.items():
        pc = sample_surface(obj.urdf)
        np.save(Path(out_dir) / f"{name}.npy", pc)


if __name__ == "__main__":
    typer.run(main)
