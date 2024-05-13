from typing import Optional, Protocol, TypedDict

import numpy as np
import numpy.typing as npt


class PlacementPointCloudData(TypedDict):
    points_action: npt.NDArray[np.float32]  # (1, num_points, 3)
    points_anchor: npt.NDArray[np.float32]  # (1, num_points, 3)
    action_symmetry_features: Optional[npt.NDArray[np.float32]]  # (1, num_points, 1)
    anchor_symmetry_features: Optional[npt.NDArray[np.float32]]  # (1, num_points, 1)
    action_symmetry_rgb: Optional[npt.NDArray[np.uint8]]  # (1, num_points, 3)
    anchor_symmetry_rgb: Optional[npt.NDArray[np.uint8]]  # (1, num_points, 3)
    phase: Optional[str]
    phase_onehot: Optional[npt.NDArray[np.float32]]  # (1, num_phases)


class PlacementPointCloudDatasetConfig(Protocol):
    dataset_type: str


class PlacementPointCloudDataset(Protocol):
    def __init__(self, cfg: PlacementPointCloudDatasetConfig):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        ...


class PretrainingPointCloudData(TypedDict):
    pass


class PretrainingPointCloudDatasetConfig(Protocol):
    dataset_type: str


class PretrainingPointCloudDataset(Protocol):
    def __init__(self, cfg: PretrainingPointCloudDatasetConfig):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> PretrainingPointCloudData:
        ...
