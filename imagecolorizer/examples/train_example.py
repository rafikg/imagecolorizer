import os

from imagecolorizer.config.config import get_cfg_defaults
from imagecolorizer.models.train import training_pipeline

config = get_cfg_defaults()
custom_config_path = os.path.join(
    config.ROOT_DIR, "config/experiments/exp0001_config.yaml"
)
training_pipeline(config_file=custom_config_path)
