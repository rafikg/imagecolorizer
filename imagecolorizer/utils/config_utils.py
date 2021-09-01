from pathlib import Path
from typing import Optional
from imagecolorizer.config.config import get_cfg_defaults
import yacs


def combine_cfgs(cfg_path: Optional[Path] = None):
    cfg_base = get_cfg_defaults()

    if cfg_path is not None and cfg_path.exists():
        cfg_base.merge_from_file(cfg_path)
    else:
        raise ValueError(f"{cfg_path} file does not exist!")
    return cfg_base


def create_experiments_dir(config: yacs.config.CfgNode):
    """
    :create_experiments_dir: Check and create the experiments folder.
    :param config: configuration object
    :type: yacs.config.CfgNode
    """
    path = Path(config.TRAIN.OUTPUT_DIR)
    if path.exists():
        raise ValueError(f"{path} is already exist !")
    else:
        # create output dir
        path.mkdir(parents=False, exist_ok=False)
