import yacs.config

from imagecolorizer.models.model import create_image_colorizer
from tensorflow.keras.models import Model


def load_model(config: yacs.config.CfgNode) -> Model:
    """
    :load_model: Load the ImageColorizer model
    :param config:
    :return:
    """
    model = create_image_colorizer()
    # Load weights inside model
    # model = tf.saved_model.load(config.TRAIN.SAVED_MODELS)
    model.load_weights(config.TRAIN.CHECKPOINT).expect_partial()
    return model
