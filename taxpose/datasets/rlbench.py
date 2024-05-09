import functools
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
from rpad.rlbench_utils.placement_dataset import AnchorMode, RLBenchPlacementDataset
from torch.utils.data import Dataset

from taxpose.datasets.augmentations import OcclusionConfig, maybe_downsample
from taxpose.datasets.base import PlacementPointCloudData


def rotational_symmetry_labels(
    points, symmetry_axis, second_axis
) -> npt.NDArray[np.float32]:
    """Get some 1-D symmetry labels for rotational symmetry.

    Args:
        points (np.ndarray): Point cloud of shape (N, 3).
        symmetry_axis (np.ndarray): Axis of symmetry of shape (3,).
        second_axis (np.ndarray): Axis which will define a vector to construct a plane.
    """
    centered_points = points - np.mean(points, axis=0, keepdims=True)

    # Compute the plane defined by the symmetry axis and the second axis.
    plane_normal = np.cross(symmetry_axis, second_axis)
    plane_normal /= np.linalg.norm(plane_normal)

    # Get the inner product between the plane normal and the centered points.
    inner_products = np.matmul(centered_points, plane_normal)

    # Normalize by the largest inner product.
    inner_products /= np.max(np.abs(inner_products))

    return inner_products.astype(np.float32)


DEMO_SYMMETRY_LABELS = {
    "stack_wine": {
        "action": {
            "symmetry_axis": "action_z",
            "second_axis": "world_z",
        },
        "anchor": None,
    },
    "insert_onto_square_peg": {
        "action": {
            "symmetry_axis": "action_z",
            "second_axis": "anchor_x",
        },
        "anchor": {
            "symmetry_axis": "anchor_z",
            # TODO: figure out if we should switch this to y.
            "second_axis": "anchor_x",
        },
    },
    "phone_on_base": None,
    "put_toilet_roll_on_stand": {
        "action": {
            "symmetry_axis": "action_z",
            "second_axis": "world_z",
        },
        "anchor": None,
    },
    "place_hanger_on_rack": None,
    "solve_puzzle": None,
    "put_knife_on_chopping_board": None,
    "reach_target": None,
    "pick_and_lift": None,
    "pick_up_cup": None,
    "put_money_in_safe": None,
    "push_button": None,
    "slide_block_to_target": None,
    "take_money_out_safe": None,
    "take_umbrella_out_of_umbrella_stand": None,
}


def get_symmetry_axes(task_name, T_action_world, T_anchor_world):
    """Get the symmetry axes from DEMO_SYMMETRY_LABELS. Hacky because of symmetry definition.

    Args:
        task_name (str): Task name
        T_action_world (np.ndarray): Transform of action point cloud to world.
        T_anchor_world (np.ndarray): Transform of anchor point cloud to world.

    Returns:
        tuple: (action_symmetry_axis, action_second_axis, anchor_symmetry_axis, anchor_second_axis)
    """

    def index_fn(axis_name):
        if "action" in axis_name:
            T = T_action_world
        elif "anchor" in axis_name:
            T = T_anchor_world
        else:
            T = np.eye(4)

        if "x" in axis_name:
            return T[:3, 0]
        elif "y" in axis_name:
            return T[:3, 1]
        elif "z" in axis_name:
            return T[:3, 2]

    if DEMO_SYMMETRY_LABELS[task_name] is None:
        return None, None, None, None

    action_symmetry_dict = DEMO_SYMMETRY_LABELS[task_name]["action"]
    if action_symmetry_dict is None:
        action_symmetry_axis = None
        action_second_axis = None
    else:
        action_symmetry_axis = index_fn(action_symmetry_dict["symmetry_axis"])
        action_second_axis = index_fn(action_symmetry_dict["second_axis"])

    anchor_symmetry_dict = DEMO_SYMMETRY_LABELS[task_name]["anchor"]
    if anchor_symmetry_dict is None:
        anchor_symmetry_axis = None
        anchor_second_axis = None
    else:
        anchor_symmetry_axis = index_fn(anchor_symmetry_dict["symmetry_axis"])
        anchor_second_axis = index_fn(anchor_symmetry_dict["second_axis"])

    return (
        action_symmetry_axis,
        action_second_axis,
        anchor_symmetry_axis,
        anchor_second_axis,
    )


def colorize_symmetry_labels(labels) -> npt.NDArray[np.uint8]:
    """Colorize the symmetry labels to smoothly interpolate between yellow and blue.

    Args:
        labels (np.ndarray): (N,) array of symmetry labels.

    Returns:
        np.ndarray: (N, 3) array of RGB colors.
    """
    # Put them on the range [-255, 255]
    color = labels / np.max(np.abs(labels)) * 255

    # Make a color array.
    color_cts = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    color_cts[labels >= 0, 0] = color[labels >= 0].astype(np.uint8)
    color_cts[labels >= 0, 1] = color[labels >= 0].astype(np.uint8)
    color_cts[labels < 0, 2] = np.abs(color[labels < 0]).astype(np.uint8)
    return color_cts


def remove_outliers_o3d(original_points):
    """Remove outliers which are far from other points using pure numpy"""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points[0])
    pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)

    # Next, cluster the points.
    labels = np.asarray(
        pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=False)
    )
    points = np.asarray(pcd.points)

    # Remove any points with -1 labels
    points = points[labels != -1]

    # Print how many were removed.
    # print(
    #     f"Removed {original_points.shape[1] - np.asarray(points).shape[0]} outliers from {original_points.shape[1]} points."
    # )
    return np.asarray(points, dtype=np.float32)[None, ...]


def remove_outliers(original_points, min_neighbors=1):
    """Remove outliers which are far from other points using pure numpy"""
    # Compute the distance between each point and every other point.
    # NOTE: this assumes that xyz's are the first 3 channels!
    dists = np.linalg.norm(
        original_points[0, None, ..., :3] - original_points[0, :, None, ..., :3], axis=-1
    )

    # The outliers are the points which are far from other points. Remove points that
    # are greater than threshold away from all other points, and ignore the diagonal.
    threshold = 0.1

    # This is really for wine stacking

    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)

    # Detect points which have fewer than min_neighbors neighbors.
    neighbors = np.sum(dists < threshold, axis=0)
    outliers = np.where(neighbors < min_neighbors)[0]

    # Remove the outliers.
    points = np.delete(original_points, outliers, axis=1)

    # Print how many were removed.
    # print(
    #     f"Removed {original_points.shape[1] - points.shape[1]} outliers from {original_points.shape[1]} points."
    # )
    return points


@dataclass
class RLBenchPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "rlbench"
    dataset_root: Path
    task_name: str = "stack_wine"
    episodes: Union[List[int], Literal["all"]] = "all"
    cached: bool = True
    phase: str = "grasp"
    use_first_as_init_keyframe: bool = True

    # Occlusion config.
    occlusion_cfg: Optional[OcclusionConfig] = None

    # Downsample the point clouds.
    num_points: Optional[int] = None

    # This allows us to use additonal ground-truth pose information to teleport
    # the initial, onoccluded observation to the final, occluded position.
    teleport_initial_to_final: bool = True
    with_symmetry: bool = True
    anchor_mode: str = "single_object"
    action_mode: str = "object"


class RLBenchPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: RLBenchPointCloudDatasetConfig):
        super().__init__()
        self.dataset = RLBenchPlacementDataset(
            dataset_root=cfg.dataset_root,
            task_name=cfg.task_name,
            demos=cfg.episodes,
            phase=cfg.phase,
            use_first_as_init_keyframe=cfg.use_first_as_init_keyframe,
            anchor_mode=cfg.anchor_mode,
            action_mode=cfg.action_mode,
        )
        # TODO: move this into configs
        self.use_rgb = True
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    @functools.lru_cache(maxsize=1000)
    def _load_data(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        data = self.dataset[index]
        # This dataset should really be about the keyframes, but there are some
        # occlusions at keyframes. We may want to switch to using an "imagined"
        # version of point clouds.
        if self.cfg.teleport_initial_to_final:
            points_action = data["init_action_pc"].numpy().astype(np.float32)
            points_anchor = data["init_anchor_pc"].numpy().astype(np.float32)

            # Teleport the initial point cloud to the final position.
            T_init_key = data["T_init_key"].numpy().astype(np.float32)

            # hstack and matmul
            points_action = np.hstack(
                [
                    points_action,
                    np.ones((points_action.shape[0], 1), dtype=np.float32),
                ]
            )
            points_action = np.matmul(points_action, T_init_key.T)[:, :3]

            points_action = points_action[None, ...]
            points_anchor = points_anchor[None, ...]

            # extracting rgb features
            rgb_action = data["init_action_rgb"].numpy().astype(np.float32) / 255.0
            rgb_anchor = data["init_anchor_rgb"].numpy().astype(np.float32) / 255.0

            # # adding rgb features
            # if self.use_rgb:
            #     action_rgb = data["init_action_rgb"].numpy().astype(np.float32) / 255.0
            #     anchor_rgb = data["init_anchor_rgb"].numpy().astype(np.float32) / 255.0
            #     points_action = np.concatenate([points_action, action_rgb[None, ...]], axis=2)
            #     points_anchor = np.concatenate([points_anchor, anchor_rgb[None, ...]], axis=2)

        else:
            points_action = data["key_action_pc"].numpy().astype(np.float32)[None, ...]
            points_anchor = data["key_anchor_pc"].numpy().astype(np.float32)[None, ...]

            # extracting rgb features
            rgb_action = data["key_action_rgb"].numpy().astype(np.float32) / 255.0
            rgb_anchor = data["key_anchor_rgb"].numpy().astype(np.float32) / 255.0
            
            # # adding rgb features
            # if self.use_rgb:
            #     action_rgb = data["key_action_rgb"].numpy().astype(np.float32) / 255.0
            #     anchor_rgb = data["key_anchor_rgb"].numpy().astype(np.float32) / 255.0
            #     points_action = np.concatenate([points_action, action_rgb[None, ...]], axis=2)
            #     points_anchor = np.concatenate([points_anchor, anchor_rgb[None, ...]], axis=2)

        rgb_action = rgb_action[None, ...]
        rgb_anchor = rgb_anchor[None, ...]

        # Quickly remove outliers...
        features_action = remove_outliers(np.concatenate([points_action, rgb_action], axis=2))
        points_action = features_action[..., :3]
        rgb_action = features_action[..., 3:]

        if (
            not self.cfg.anchor_mode == AnchorMode.RAW
            and not self.cfg.anchor_mode == AnchorMode.BACKGROUND_ROBOT_REMOVED
        ):
            features_anchor = remove_outliers(np.concatenate([points_anchor, rgb_anchor], axis=2))
            points_anchor = features_anchor[..., :3]
            rgb_anchor = features_anchor[..., 3:]

        new_data = {
            "T_action_key_world": data["T_action_key_world"],
            "T_anchor_key_world": data["T_anchor_key_world"],
            "phase": data["phase"],
            "phase_onehot": data["phase_onehot"],
            "rgb_action": rgb_action,
            "rgb_anchor": rgb_anchor,
        }
        return points_action, points_anchor, new_data

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        # Load the data.
        points_action, points_anchor, data = self._load_data(index)
        # separate rgb features
        # if self.use_rgb:
        #     points_action, action_rgb = points_action[..., :3], points_action[..., 3:]
        #     points_anchor, anchor_rgb = points_anchor[..., :3], points_anchor[..., 3:]
        rgb_action = data["rgb_action"]
        rgb_anchor = data["rgb_anchor"]

        # Occlude if necessary.
        # TODO: implement occlusions here.

        # Downsample if necessary.
        try:
            points_action, action_ids = maybe_downsample(points_action, self.cfg.num_points)
            rgb_action = rgb_action[0, action_ids]

            num_anchor_points = (
                self.cfg.num_points
                if not self.cfg.anchor_mode == AnchorMode.RAW
                else 1024
            )
            points_anchor, anchor_ids = maybe_downsample(points_anchor, num_anchor_points)
            rgb_anchor = rgb_anchor[0, anchor_ids]
        except:
            print(f"Failed to downsample {index}.")

        if self.cfg.with_symmetry:
            T_action_key_world = data["T_action_key_world"].numpy().astype(np.float32)
            T_anchor_key_world = data["T_anchor_key_world"].numpy().astype(np.float32)

            # Get the symmetry axes.
            (
                action_symmetry_axis,
                action_second_axis,
                anchor_symmetry_axis,
                anchor_second_axis,
            ) = get_symmetry_axes(
                self.cfg.task_name, T_action_key_world, T_anchor_key_world
            )

            # Z is axis of rotation, x is the plane.
            if action_symmetry_axis is None:
                action_symmetry_features = np.ones(
                    (1, points_action.shape[1], 1), dtype=np.float32
                )
            else:
                action_symmetry_features = rotational_symmetry_labels(
                    points_action[0], action_symmetry_axis, action_second_axis
                )[None, :, None]

            if anchor_symmetry_axis is None:
                anchor_symmetry_features = np.ones(
                    (1, points_anchor.shape[1], 1), dtype=np.float32
                )
            else:
                anchor_symmetry_features = rotational_symmetry_labels(
                    points_anchor[0], anchor_symmetry_axis, anchor_second_axis
                )[None, :, None]

        else:
            # For now, we need to hack in the symmetry features. We'll almost certainly
            # want to do this differently in the future.
            action_symmetry_features = np.ones(
                (1, points_action.shape[0], 1), dtype=np.float32
            )
            anchor_symmetry_features = np.ones(
                (1, points_anchor.shape[0], 1), dtype=np.float32
            )

        # Colors.
        action_symmetry_rgb = colorize_symmetry_labels(
            action_symmetry_features[0, ..., 0]
        )[None]
        anchor_symmetry_rgb = colorize_symmetry_labels(
            anchor_symmetry_features[0, ..., 0]
        )[None]

        res = {}
        # Assert shapes
        assert len(points_action.shape) == 3
        assert len(points_anchor.shape) == 3
        assert len(action_symmetry_features.shape) == 3
        assert len(anchor_symmetry_features.shape) == 3
        assert len(action_symmetry_rgb.shape) == 3
        assert len(anchor_symmetry_rgb.shape) == 3
        assert len(rgb_action.shape) == 3
        assert len(rgb_anchor.shape) == 3

        # assert rgb features shapes
        # if self.use_rgb:
        #     assert len(rgb.shape) == 3
        #     assert len(anchor_rgb.shape) == 3
        #     res.update({
        #         "rgb_action": action_rgb,
        #         "rgb_anchor": anchor_rgb,
        #     })

        res.update({
            "points_action": points_action,
            "points_anchor": points_anchor,
            "rgb_action": rgb_action,
            "rgb_anchor": rgb_anchor,
            "action_symmetry_features": action_symmetry_features,
            "anchor_symmetry_features": anchor_symmetry_features,
            "action_symmetry_rgb": action_symmetry_rgb,
            "anchor_symmetry_rgb": anchor_symmetry_rgb,
            "phase": data["phase"],
            "phase_onehot": torch.as_tensor(data["phase_onehot"]),
        })
        return res
