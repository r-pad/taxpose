# Much around with the path to make the import work
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

# Add the parent directory to the path to import the script. Hacky, but it works.
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

from scripts.train_residual_flow import main


def _get_training_config_names(bmark, ablation=False):
    # Get config paths from the configs/commands directory, relative to the commands directory.
    configs = [path for path in Path(f"configs/commands/{bmark}").rglob("*.yaml")]

    # Strip the "configs/" prefix.
    configs = [str(path)[8:] for path in configs]

    # Filter out paths with basenames that have a leading underscore.
    configs = [config for config in configs if not Path(config).name.startswith("_")]

    # Filter out paths that don't include the word "train" in the path.
    configs = [config for config in configs if "train_" in config]

    if ablation:
        # Filter out paths that don't include the word "ablation" in the path except for
        configs = [config for config in configs if "ablation" in config]
    else:
        # Filter out paths that include the word "ablation" in the path.
        configs = [config for config in configs if "ablation" not in config]

    # Filter out paths with any folder that have a leading underscore.
    configs = [
        config
        for config in configs
        if not any(folder.startswith("_") for folder in Path(config).parts)
    ]

    return configs


DEFAULT_DATA_PATH = "/data"
DEFAULT_RLBENCH_ROOT = "rlbench"


def _test_commands_run(config_name, bmark_dataset_root="ndf"):
    dataset_root = (
        os.environ["DEFAULT_DATA_PATH"]
        if "DEFAULT_DATA_PATH" in os.environ
        else DEFAULT_DATA_PATH
    )
    bmark_dataset_root = os.path.join(dataset_root, bmark_dataset_root)
    torch.multiprocessing.set_sharing_strategy("file_system")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.runtime.output_dir=.",
                "seed=1234",
                f"data_root={dataset_root}",
                f"benchmark.dataset_root={bmark_dataset_root}",
                "training.batch_size=2",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        # Just for this function call, set the environment variable to WANDB_MODE=disabled
        os.environ["WANDB_MODE"] = "disabled"
        # Run the training script.
        main(cfg)


# Skip this if the environment variable is not set or the path does not exist.
@pytest.mark.ndf_training
@pytest.mark.skipif(
    ("DEFAULT_DATA_PATH" not in os.environ or not os.path.exists(DEFAULT_DATA_PATH))
    and not torch.cuda.is_available(),
    reason="DEFAULT_DATA_PATH environment variable is not set or the path does not exist.",
)
@pytest.mark.parametrize("config_name", _get_training_config_names("ndf"))
def test_ndf_training_commands_run(config_name):
    _test_commands_run(config_name, "ndf")


# Do the same for the ablation configs.
@pytest.mark.ablations
@pytest.mark.skipif(
    ("DEFAULT_DATA_PATH" not in os.environ or not os.path.exists(DEFAULT_DATA_PATH))
    and not torch.cuda.is_available(),
    reason="DEFAULT_DATA_PATH environment variable is not set or the path does not exist.",
)
@pytest.mark.parametrize(
    "config_name", _get_training_config_names("ndf", ablation=True)
)
def test_training_ablation_commands_run(config_name):
    _test_commands_run(config_name, "ndf")


# Skip this if the environment variable is not set or the path does not exist.
@pytest.mark.rlbench_training
@pytest.mark.parametrize("config_name", _get_training_config_names("rlbench"))
def test_rlbench_training_commands_run(config_name):
    _test_commands_run(config_name, DEFAULT_RLBENCH_ROOT)
