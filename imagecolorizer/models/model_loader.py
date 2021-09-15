import yacs.config
from tensorflow.keras.models import Model

from imagecolorizer.models.model import create_image_colorizer


def load_model(*, config: yacs.config.CfgNode) -> Model:
    """
    :load_model: Load the ImageColorizer model
    :param config:
    :return:
    """
    model = create_image_colorizer()
    # Load weights inside model
    model.load_weights(config.TEST.TRAINED_MODEL)
    return model
