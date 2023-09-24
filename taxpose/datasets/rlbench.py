import functools
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from rpad.rlbench_utils.placement_dataset import RLBenchPlacementDataset
from torch.utils.data import Dataset

from taxpose.datasets.base import PlacementPointCloudData


@dataclass
class RLBenchPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "rlbench"
    dataset_root: Path
    task_name: str = "stack_wine"
    n_demos: int = 10
    start_ix: int = 0
    cached: bool = True


class RLBenchPlacementDatasetCached(Dataset[PlacementPointCloudData]):
    def __init__(
        self, dataset_root: Path, task_name: str, n_demos: int, start_ix: int = 0
    ):
        super().__init__()

        self.files = [
            Path(str(dataset_root) + "_processed") / task_name / f"episode{i}.npz"
            for i in range(start_ix, start_ix + n_demos)
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
            "action_symmetry_features": np.ones(
                (1, data["action_pc"].shape[1], 1), dtype=np.float32
            ),
            "anchor_symmetry_features": np.ones(
                (1, data["anchor_pc"].shape[1], 1), dtype=np.float32
            ),
            "anchor_symmetry_rgb": np.ones(
                (1, data["anchor_pc"].shape[1], 3), dtype=np.float32
            ),
            "action_symmetry_rgb": np.ones(
                (1, data["action_pc"].shape[1], 3), dtype=np.float32
            ),
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
                start_ix=cfg.start_ix,
            )
        else:
            self.dataset = RLBenchPlacementDataset(
                dataset_root=cfg.dataset_root,
                task_name=cfg.task_name,
                n_demos=cfg.n_demos,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> PlacementPointCloudData:
        data = self.dataset[index]

        return {
            "points_action": data["action_pc"].numpy().astype(np.float32)[None, ...],
            "points_anchor": data["anchor_pc"].numpy().astype(np.float32)[None, ...],
            # "symmetric_cls": np.asarray([], dtype=np.float32),
            "action_symmetry_features": np.ones(
                (1, data["action_pc"].shape[0], 1), dtype=np.float32
            ),
            "anchor_symmetry_features": np.ones(
                (1, data["anchor_pc"].shape[0], 1), dtype=np.float32
            ),
            "anchor_symmetry_rgb": np.ones(
                (1, data["anchor_pc"].shape[0], 3), dtype=np.float32
            ),
            "action_symmetry_rgb": np.ones(
                (1, data["action_pc"].shape[0], 3), dtype=np.float32
            ),
        }
