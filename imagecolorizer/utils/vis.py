import matplotlib.pyplot as plt
import numpy as np


def display_result(origin_image: np.ndarray, result_image: np.ndarray) -> None:
    """
    Display the colorizing result
    :param origin_image: Gray Scale image
    :type origin_image: np.ndarray

    :param result_image: RGB image
    :type result_image: np.ndarray
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Colorizing result")
    ax1.imshow(origin_image)
    ax1.set_title("Original image")
    ax2.imshow(result_image)
    ax2.set_title("RGB image")
    plt.show()
