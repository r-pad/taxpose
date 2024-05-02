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

from scripts.pretrain_embedding import main


def _get_pretraining_config_names():
    # Get config paths from the configs/commands directory, relative to the commands directory.
    configs = [
        path for path in Path(f"configs/commands/ndf/pretraining/").rglob("*.yaml")
    ]

    # Strip the "configs/" prefix.
    configs = [str(path)[8:] for path in configs]

    # Filter out paths with basenames that have a leading underscore.
    configs = [config for config in configs if not Path(config).name.startswith("_")]

    # Filter out paths with any folder that have a leading underscore.
    configs = [
        config
        for config in configs
        if not any(folder.startswith("_") for folder in Path(config).parts)
    ]

    return configs


DEFAULT_NDF_PATH = "/data"


@pytest.mark.skipif(
    ("NDF_DATASET_ROOT" not in os.environ or not os.path.exists(DEFAULT_NDF_PATH))
    and not torch.cuda.is_available(),
    reason="NDF_DATASET_ROOT environment variable is not set or the path does not exist.",
)
@pytest.mark.parametrize("config_name", _get_pretraining_config_names())
def test_pretraining(config_name):
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
                f"data_root={dataset_root}",
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
