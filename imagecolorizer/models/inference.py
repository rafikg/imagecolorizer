from typing import Dict, Optional

import numpy as np
from skimage.color import lab2rgb, rgb2lab

from imagecolorizer.models.model_loader import load_model
from imagecolorizer.utils.config_utils import combine_cfgs
from imagecolorizer.utils.logger import CustomLogger
from imagecolorizer.utils.random_seed import set_random_seeds
from imagecolorizer.utils.validation import valid_input_data

logger = CustomLogger(name=__name__).get_logger()

set_random_seeds()


def make_predictions(*, input_data: Dict, config_file: Optional[str]) -> Dict:
    """
    :make_predictions: run the inference pipeline on an input_data
    :param input_data: input data as float values in [0, 255]
    :type input_data: dict

    :param config_file: custom configuration file
    :type config_file: str
    :return:
        dict of prediction output.
    """
    # Combine config_file with default config
    config = combine_cfgs(cfg_path=config_file)
    logger.debug(f"Test config: \n {config}")
    image = input_data["instances"]

    image = valid_input_data(image=image, config=config)

    # Prepare data
    lab = rgb2lab(np.array(image) / 255.0)
    l_channel = lab[:, :, 0]
    l_rgb = np.stack((l_channel,) * 3)
    l_rgb = l_rgb.reshape((1, *config.DATA.INPUT_SIZE, 3))

    # Load model
    model = load_model(config=config)

    # Make prediction
    ab = model.predict(l_rgb)
    ab = ab * 128
    color_img = np.zeros((*config.DATA.INPUT_SIZE, 3))

    color_img[:, :, 0] = l_channel
    color_img[:, :, 1:] = ab
    return {"predictions": lab2rgb(color_img)}
