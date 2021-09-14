import yacs.config
import numpy as np
import cv2


def valid_input_data(image: np.ndarray, config: yacs.config.CfgNode):
    """
    Validate input data
    :param image: grayscale image
    :type image ndaaray

    :param config: configuration file
    :type config: yacs.config
    :return: ndarray
    """
    if image.ndim != 1:
        raise ValueError('input image should be grayscale')
    if image.shape[0] != config.INPUT_SIZE[0] or image.shape[1] != \
            config.INPUT_SIZE[1]:
        image = cv2.resize(src=image, dsize=config.INPUT_SIZE)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image
