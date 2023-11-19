from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np
from rpad.rlbench_utils.placement_dataset import RLBenchPlacementDataset, StackWinePhase
from torch.utils.data import Dataset

from taxpose.datasets.base import PlacementPointCloudData


@dataclass
class RLBenchPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "rlbench"
    dataset_root: Path
    task_name: str = "stack_wine"
    episodes: Union[List[int], Literal["all"]] = "all"
    cached: bool = True
    phase: StackWinePhase = "grasp"


class RLBenchPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: RLBenchPointCloudDatasetConfig):
        super().__init__()

        self.dataset = RLBenchPlacementDataset(
            dataset_root=cfg.dataset_root,
            task_name=cfg.task_name,
            n_demos=-1,
            phase=cfg.phase,
        )

        # Induce a subset if necessary.
        self.episode_ixs = (
            list(range(len(self.dataset))) if cfg.episodes == "all" else cfg.episodes
        )

    def __len__(self):
        return len(self.episode_ixs)

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        data = self.dataset[self.episode_ixs[index]]

        # This dataset should really be about the keyframes, but there are some
        # occlusions at keyframes. We may want to switch to using an "imagined"
        # version of point clouds.
        points_action = data["key_action_pc"].numpy().astype(np.float32)[None, ...]
        points_anchor = data["key_anchor_pc"].numpy().astype(np.float32)[None, ...]

        # For now, we need to hack in the symmetry features. We'll almost certainly
        # want to do this differently in the future.
        action_symmetry_features = np.ones(
            (1, points_action.shape[0], 1), dtype=np.float32
        )
        anchor_symmetry_features = np.ones(
            (1, points_anchor.shape[0], 1), dtype=np.float32
        )

        # Assert shapes
        assert len(points_action.shape) == 3
        assert len(points_anchor.shape) == 3
        assert len(action_symmetry_features.shape) == 3
        assert len(anchor_symmetry_features.shape) == 3

        return {
            "points_action": points_action,
            "points_anchor": points_anchor,
            "action_symmetry_features": action_symmetry_features,
            "anchor_symmetry_features": anchor_symmetry_features,
        }
