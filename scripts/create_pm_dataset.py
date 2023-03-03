from enum import Enum
from pathlib import Path

import typer

from taxpose.datasets.pm_placement import (
    GoalTransferDataset,
    PlaceDataset,
    scenes_by_location,
)


class Split(str, Enum):
    train = "train"
    test = "test"
    unseen = "unseen"


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
    n_proc: int = 32,
    randomize_camera: bool = True,
    rotate_anchor: bool = True,
):
    if goal_location == "all":
        locs = ["in", "top", "left", "right", "under"]
    else:
        locs = [goal_location]

    obs_scene_ids = []
    goal_scene_ids = []

    for loc in locs:
        otr = scenes_by_location(split.value, "obs", loc.value)
        gtr = scenes_by_location(split.value, "goal", loc.value)

        obs_scene_ids.extend([(t[0], t[1], t[2], loc.value) for t in otr])
        goal_scene_ids.extend([(t[0], t[1], t[2], loc.value) for t in gtr])

    dset = GoalTransferDataset(
        obs_dset=PlaceDataset(
            pm_root,
            scene_ids=obs_scene_ids,
            use_processed=True,
            n_repeat=n_repeat,
            mode="obs",
            randomize_camera=randomize_camera,
            n_proc=n_proc,
            snap_to_surface=True,
            full_obj=True,
            even_downsample=True,
        ),
        goal_dset=PlaceDataset(
            pm_root,
            scene_ids=goal_scene_ids,
            use_processed=True,
            n_repeat=n_repeat,
            mode="goal",
            randomize_camera=randomize_camera,
            n_proc=n_proc,
            snap_to_surface=True,
            full_obj=True,
            even_downsample=True,
        ),
        rotate_anchor=rotate_anchor,
    )


if __name__ == "__main__":
    typer.run(main)
