from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _get_config_names():
    # Get config paths from the configs/commands directory, relative to the commands directory.
    configs = [path for path in Path("configs/commands").rglob("*.yaml")]

    # Strip the "configs/" prefix.
    configs = [str(path)[8:] for path in configs]

    # Filter out paths with basenames that have a leading underscore.
    configs = [config for config in configs if not Path(config).name.startswith("_")]

    return configs


HYDRA_CONFIG = {}


@pytest.mark.parametrize("config_name", _get_config_names())
def test_commands_compile(config_name):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.runtime.output_dir=.",
                "seed=1234",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        # Resolve to yaml.
        yaml_cfg = OmegaConf.to_yaml(cfg, resolve=True)

        assert cfg.job_type is not None
        assert cfg.wandb.save_dir is not None
