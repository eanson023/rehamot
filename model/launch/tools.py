import os

from omegaconf import DictConfig


def resolve_cfg_path(cfg: DictConfig):
    working_dir = os.getcwd()
    cfg.working_dir = working_dir
