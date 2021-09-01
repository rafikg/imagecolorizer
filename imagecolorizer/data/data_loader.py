from pathlib import Path
from typing import Tuple, List
import numpy as np
import glob
import os

from tensorflow.keras.utils import Sequence
from imagecolorizer.utils.logger import CustomLogger
from imagecolorizer.utils.preprocess import get_lab_channels


logger = CustomLogger(name=__name__).get_logger()


class GrayScaleData(Sequence):
    """
    GrayScaleData class to hold sequence of image.
    """

    def __init__(self, images_paths: List,
                 batch_size: int = 16,
                 target_dim: Tuple = (224, 224),
                 shuffle: bool = False):
        self.images_paths = images_paths
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Number of batch in the sequence

        :return:
            The number of batch in the sequence
        """
        return int(np.ceil(len(self.images_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Gets batch at position `index`
        :param index: position of the batch in the sequence
        :type index: int

        :return:
            A batch
        """
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        X = np.array([np.stack(
            (get_lab_channels(self.images_paths[k])[:, :, 0],) * 3,
            axis=-1) for k in indexes])
        y = np.array(
            [get_lab_channels(self.images_paths[k])[:, :, 1:] / 128. for k
             in indexes])
        return X, y

    def on_epoch_end(self):
        """
        Method called at the end of every epoch to shuffle the images indexes
        """
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def validate_and_load_data(image_folder_path: str,
                           exts_list: Tuple = ('*.jpg', '*.png', '*.jpeg')):
    """
    :validate_and_load_data: validate and load images pathes.

    :param image_folder_path: folder path
    :type image_folder_path: Path

    :param exts_list: possible image extensions
    :type exts_list: Tuple

    :return:
    """
    grabbed_files = []
    if Path(image_folder_path).exists():
        for files in exts_list:
            grabbed_files.extend(glob.glob(os.path.join(image_folder_path,
                                                        files)))

    else:
        raise ValueError(f'{image_folder_path} does not exist')
    logger.info(f'...Loading {len(grabbed_files)} images')
    return grabbed_files
