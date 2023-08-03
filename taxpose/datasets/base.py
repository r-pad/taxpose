from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt


class PlacementPointCloudData(TypedDict):
    points_action: npt.NDArray[np.float32]  # (1, num_points, 3)
    points_anchor: npt.NDArray[np.float32]  # (1, num_points, 3)
    symmetric_cls: npt.NDArray[np.float32]  # Not really sure what this is...


class PlacementPointCloudDatasetConfig(Protocol):
    dataset_type: str


class PlacementPointCloudDataset(Protocol):
    def __init__(self, cfg: PlacementPointCloudDatasetConfig):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        ...
