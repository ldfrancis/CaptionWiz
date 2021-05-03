from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import wandb
import yaml

from captionwiz.utils.type import Dir, FilePath


def load_yaml(file_path: FilePath) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


def get_captionwiz_dir() -> Dir:
    captionwiz_dir = Path.home() / "captionwiz"
    return captionwiz_dir


def get_datetime(format="%Y%m%d%H%M%S") -> str:
    now = datetime.now()
    dtime = now.strftime(format)
    return dtime


def setup_wandb(cfg):
    if cfg["use_wandb"]:
        project = cfg["wandb"]["project"]
        wandb.init(project=project, sync_tensorboard=True, config=cfg)
