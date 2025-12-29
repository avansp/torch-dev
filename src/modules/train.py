import rootutils
import hydra
from omegaconf import DictConfig
import time
import logging
import lightning as lit
from typing import List
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from modules.utils import (
    console,
    custom_log,
    instantiators,
    metrics
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    """
    Main entry point for training
    
    :param cfg: A DictConfig configuration composed by Hydra
    """
    # log the time starting the train
    start_training = time.process_time()
    logging.info(f"Train {cfg.name=} started.")

    # print config
    if cfg.print_config:
        console.print_config_tree(cfg, resolve=True, save_to_file=True, save_to_yaml=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        logging.info(f"Setting random seed to {cfg.seed}.")
        lit.seed_everything(cfg.seed, workers=True)

    # INSTANTIATIONS

    logging.info(f"Instantiating model <{cfg.model._target_}>")
    model: lit.LightningModule = hydra.utils.instantiate(cfg.model)

    logging.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: lit.LightningDataModule = hydra.utils.instantiate(cfg.data)

    logging.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))

    logging.info("Instantiating loggers...")
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    logging.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # SAVE HPARAMS

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        logging.info("Logging hyperparameters!")
        custom_log.log_hyperparameters(object_dict)

    # START TRAINING
    logging.info("Start training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_result = trainer.callback_metrics
    logging.info(f"Last train result = {train_result!r}")

    # FINISH TRAINING
    logging.success(f"Train {cfg.name=} finished, elapsed {time.process_time() - start_training} sec.")
    logging.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    train()
