import cv2
import numpy as np
import yacs.config


def valid_input_data(*, image: np.ndarray, config: yacs.config.CfgNode) -> np.ndarray:
    """
    Validate input data
    :param image: grayscale image
    :type image ndaaray

    :param config: configuration file
    :type config: yacs.config
    :return: ndarray
    """
    if (
        image.shape[0] != config.DATA.INPUT_SIZE[0]
        or image.shape[1] != config.DATA.INPUT_SIZE[1]
    ):
        image = cv2.resize(src=image, dsize=config.DATA.INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image
