from pathlib import Path

import numpy as np
import typer
from rpad.rlbench_utils.placement_dataset import RLBenchPlacementDataset


def main(
    rlbench_root: Path = typer.Argument(..., exists=True, file_okay=False),
    out_dir: Path = typer.Argument(..., file_okay=False),
    task_name: str = "stack_wine",
    phase: str = "grasp",
):
    out_task_dir = out_dir / task_name / phase
    if not out_task_dir.exists():
        out_task_dir.mkdir(parents=True)

    # Count the number of directories in the task folder:
    task_folder = rlbench_root / task_name / "variation0/episodes"
    n_files = len(list(task_folder.iterdir()))

    dset = RLBenchPlacementDataset(
        dataset_root=str(rlbench_root),
        task_name=task_name,
        n_demos=n_files,
        phase=phase,
    )

    # Iterate over the dataset.
    for i in range(len(dset)):
        data = dset[i]

        # Save the data to a file.
        np.savez(
            out_task_dir / f"episode{i}.npz",
            **{k: v.numpy().astype(np.float32) for k, v in data.items()},
        )


if __name__ == "__main__":
    typer.run(main)
