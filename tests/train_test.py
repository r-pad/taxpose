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
from scripts.train_residual_flow_ablation import main as main_ablation


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
        configs = [
            config
            for config in configs
            if "ablation" in config and "n_demo" not in config
        ]
    else:
        # Filter out paths that include the word "ablation" in the path.
        configs = [
            config
            for config in configs
            if "ablation" not in config or "n_demo" in config
        ]

    # Filter out paths with any folder that have a leading underscore.
    configs = [
        config
        for config in configs
        if not any(folder.startswith("_") for folder in Path(config).parts)
    ]

    return configs


DEFAULT_NDF_PATH = "/data/ndf"


# Skip this if the environment variable is not set or the path does not exist.
@pytest.mark.training
@pytest.mark.skipif(
    ("NDF_DATASET_ROOT" not in os.environ or not os.path.exists(DEFAULT_NDF_PATH))
    and not torch.cuda.is_available(),
    reason="NDF_DATASET_ROOT environment variable is not set or the path does not exist.",
)
@pytest.mark.parametrize("config_name", _get_training_config_names("ndf"))
def test_training_commands_run(config_name):
    dataset_root = (
        os.environ["NDF_DATASET_ROOT"]
        if "NDF_DATASET_ROOT" in os.environ
        else DEFAULT_NDF_PATH
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.runtime.output_dir=.",
                "seed=1234",
                f"dataset_root={dataset_root}",
                "batch_size=2",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        # Just for this function call, set the environment variable to WANDB_MODE=disabled
        os.environ["WANDB_MODE"] = "disabled"
        # Run the training script.
        main(cfg)


# Do the same for the ablation configs.
@pytest.mark.ablations
@pytest.mark.skipif(
    ("NDF_DATASET_ROOT" not in os.environ or not os.path.exists(DEFAULT_NDF_PATH))
    and not torch.cuda.is_available(),
    reason="NDF_DATASET_ROOT environment variable is not set or the path does not exist.",
)
@pytest.mark.parametrize(
    "config_name", _get_training_config_names("ndf", ablation=True)
)
def test_training_ablation_commands_run(config_name):
    dataset_root = (
        os.environ["NDF_DATASET_ROOT"]
        if "NDF_DATASET_ROOT" in os.environ
        else DEFAULT_NDF_PATH
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.runtime.output_dir=.",
                "seed=1234",
                f"dataset_root={dataset_root}",
                "batch_size=2",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        # Just for this function call, set the environment variable to WANDB_MODE=disabled
        os.environ["WANDB_MODE"] = "disabled"
        # Run the training script.
        main_ablation(cfg)
