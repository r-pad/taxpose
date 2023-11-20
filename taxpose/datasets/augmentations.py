from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.ops import sample_farthest_points

from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion


def maybe_downsample(
    points: npt.NDArray[np.float32], num_points: Optional[int] = None
) -> npt.NDArray[np.float32]:
    if num_points is None:
        return points

    if points.shape[1] < num_points:
        raise ValueError("Cannot downsample to more points than exist in the cloud.")

    points, ids = sample_farthest_points(
        torch.from_numpy(points), K=num_points, random_start_point=True
    )

    return points.numpy()


@dataclass
class OcclusionConfig:
    occlusion_class: int

    # Ball occlusion.
    ball_occlusion: bool = True
    ball_radius: float = 0.1

    # Plane occlusion.
    plane_occlusion: bool = True
    plane_standoff: float = 0.04

    occlusion_prob: float = 0.5


def occlusion_fn(
    cfg: Optional[OcclusionConfig] = None,
) -> Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]:
    if cfg is None:
        return lambda x, y, z: x

    def occlusion(points: npt.NDArray[np.float32], obj_class: int, min_num_points: int):
        if obj_class == cfg.occlusion_class:
            if cfg.ball_occlusion:
                if np.random.rand() > cfg.occlusion_prob:
                    points_new, _ = ball_occlusion(points[0], radius=cfg.ball_radius)

                    # Ignore the occlusion if it's going to mess us up later...
                    if points_new.shape[0] > min_num_points:
                        points = points_new.unsqueeze(0)

            if cfg.plane_occlusion:
                if np.random.rand() > cfg.occlusion_prob:
                    points_new, _ = plane_occlusion(
                        points[0], stand_off=cfg.plane_standoff
                    )
                    # Ignore the occlusion if it's going to mess us up later...
                    if points_new.shape[0] > min_num_points:
                        points = points_new.unsqueeze(0)
        return points

    return occlusion
