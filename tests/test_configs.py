from omegaconf import DictConfig
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

def test_train_config(cfg_train: DictConfig):
    """Unit test for train.yaml"""
    assert cfg_train
    assert cfg_train.name
    assert cfg_train.seed

    HydraConfig().set_config(cfg_train)

