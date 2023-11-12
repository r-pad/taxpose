import functools
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from rpad.rlbench_utils.placement_dataset import RLBenchPlacementDataset, StackWinePhase
from torch.utils.data import Dataset

from taxpose.datasets.base import PlacementPointCloudData


@dataclass
class RLBenchPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "rlbench"
    dataset_root: Path
    task_name: str = "stack_wine"
    n_demos: int = 10
    cached: bool = True
    phase: StackWinePhase = "grasp"


class RLBenchPlacementDatasetCached(Dataset[PlacementPointCloudData]):
    def __init__(
        self, dataset_root: Path, task_name: str, n_demos: int, phase: StackWinePhase
    ):
        super().__init__()

        self.files = [
            Path(str(dataset_root) + "_processed")
            / task_name
            / phase
            / f"episode{i}.npz"
            for i in range(n_demos)
        ]

        # Check that all files exist.
        for f in self.files:
            if not f.exists():
                raise FileNotFoundError(f"File {f} does not exist.")

    @functools.cache
    def __getitem__(self, index) -> PlacementPointCloudData:
        data = np.load(self.files[index])

        return {
            "action_pc": torch.as_tensor(data["action_pc"]),
            "anchor_pc": torch.as_tensor(data["anchor_pc"]),
        }

    def __len__(self):
        return len(self.files)


class RLBenchPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: RLBenchPointCloudDatasetConfig):
        super().__init__()

        if cfg.cached:
            self.dataset = RLBenchPlacementDatasetCached(
                dataset_root=cfg.dataset_root,
                task_name=cfg.task_name,
                n_demos=cfg.n_demos,
                phase=cfg.phase,
            )
        else:
            self.dataset = RLBenchPlacementDataset(
                dataset_root=cfg.dataset_root,
                task_name=cfg.task_name,
                n_demos=cfg.n_demos,
                phase=cfg.phase,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> PlacementPointCloudData:
        data = self.dataset[index]

        return {
            "points_action": data["action_pc"].numpy().astype(np.float32)[None, ...],
            "points_anchor": data["anchor_pc"].numpy().astype(np.float32)[None, ...],
            "symmetric_cls": np.asarray([], dtype=np.float32),
        }
