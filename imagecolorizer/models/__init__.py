import tensorflow as tf

from imagecolorizer.utils.logger import CustomLogger

logger = CustomLogger(name=__name__).get_logger()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) >= 1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logger.info('Machine does not contains GPU')
