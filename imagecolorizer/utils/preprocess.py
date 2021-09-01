from typing import Tuple

import cv2
import skimage.io
from skimage.color import rgb2lab


def get_lab_channels(path: str, dsize: Tuple = (224, 224)):
    """
    :get_lab_channels: convert rgb image to lab image
    :param path: image path
    :type: str
    :param dsize: image's target size
    :type dsize: Tuple
    :return:
    lab image
    """
    image = cv2.resize(skimage.io.imread(path),
                       dsize=dsize,
                       interpolation=cv2.INTER_CUBIC)

    if image.ndim != 3:
        raise ValueError("Error grayscale image")

    image = image / 255.
    lab = rgb2lab(image)

    return lab
