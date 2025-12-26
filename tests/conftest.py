"""This file prepares files and configurations for unit tests."""

import pytest
from omegaconf import DictConfig, open_dict
from hydra import compose, initialize
import rootutils
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(scope="package")
def cfg_train_minimal() -> DictConfig:
    """A pytest fixture to setup a minimal running train configuration."""
    root_dir = rootutils.find_root(indicator=".project-root")

    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])
        cfg.paths.root_dir = str(root_dir)

    return cfg

@pytest.fixture(scope="function")
def cfg_train(cfg_train_minimal: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_minimal()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.
    """
    cfg = cfg_train_minimal.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
