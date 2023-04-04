import os
from enum import Enum
from pathlib import Path

import typer
from rpad.pyg.dataset import CachedByKeyDataset

from taxpose.datasets.pm_placement import PlaceDataset, scenes_by_location


class Split(str, Enum):
    train = "train"
    test = "test"


class GoalLocation(str, Enum):
    inside = "in"
    top = "top"
    left = "left"
    right = "right"
    under = "under"
    all_locs = "all"


def main(
    pm_root: Path = typer.Option(..., dir_okay=True, file_okay=False),
    split: Split = Split.train,
    goal_location: GoalLocation = GoalLocation.all_locs,
    n_repeat: int = 100,
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
    randomize_camera: bool = True,
    rotate_anchor: bool = True,
    snap_to_surface: bool = True,
    full_obj: bool = True,
    even_downsample: bool = True,
    seed: int = 123456,
):
    # This is so we can properly distribute.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if goal_location == "all":
        locs = ["in", "top", "left", "right", "under"]
    else:
        locs = [goal_location]

    obs_scene_ids = []
    goal_scene_ids = []

    for loc in locs:
        otr = scenes_by_location(split, "obs", loc)
        gtr = scenes_by_location(split, "goal", loc)

        obs_scene_ids.extend([(t[0], t[1], t[2], loc) for t in otr])
        goal_scene_ids.extend([(t[0], t[1], t[2], loc) for t in gtr])

    obs_scene_ids = sorted(obs_scene_ids)
    goal_scene_ids = sorted(goal_scene_ids)

    dset_kwargs = dict(
        root=pm_root,
        randomize_camera=randomize_camera,
        snap_to_surface=snap_to_surface,
        full_obj=full_obj,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    obs_dset = CachedByKeyDataset(
        dset_cls=PlaceDataset,
        dset_kwargs={
            **dset_kwargs,
            **{
                "scene_ids": obs_scene_ids,
                "mode": "obs",
            },
        },
        data_keys=obs_scene_ids,
        root=pm_root,
        processed_dirname=PlaceDataset.processed_dir_name(
            "obs",
            randomize_camera,
            snap_to_surface,
            full_obj,
            even_downsample,
        ),
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=123456,
    )

    goal_dset = CachedByKeyDataset(
        dset_cls=PlaceDataset,
        dset_kwargs={
            **dset_kwargs,
            **{
                "scene_ids": goal_scene_ids,
                "mode": "goal",
            },
        },
        data_keys=goal_scene_ids,
        root=pm_root,
        processed_dirname=PlaceDataset.processed_dir_name(
            "goal",
            randomize_camera,
            snap_to_surface,
            full_obj,
            even_downsample,
        ),
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=654321,
    )


if __name__ == "__main__":
    typer.run(main)
