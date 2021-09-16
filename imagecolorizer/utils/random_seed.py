import os
import random

import numpy as np
import tensorflow as tf


def set_random_seeds() -> None:
    """
    fix all random seeds
    :return:
    """
    # set env variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(123)

    # numpy seed
    np.random.seed(123)

    # tensorflow seed
    tf.random.set_seed(123)

    # python seed
    random.seed(123)
