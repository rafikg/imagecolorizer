from pathlib import Path

import cv2
import numpy as np
import yacs.config

from imagecolorizer.utils.config_utils import combine_cfgs

config = combine_cfgs()


def load_image_sample(*, config: yacs.config.CfgNode) -> np.ndarray:
    if Path(config.TEST.IMAGE_SAMPLE_PATH).exists():
        image = cv2.imread(config.TEST.IMAGE_SAMPLE_PATH, 0)
    else:
        raise ValueError(f"{config.TEST.IMAGE_SAMPLE_PATH} does not exist!")
    return image
