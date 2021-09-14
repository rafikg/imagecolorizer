from pathlib import Path
from typing import Dict, Optional
from skimage.color import rgb2lab
from imagecolorizer.models.model_loader import load_model
from imagecolorizer.utils.config_utils import combine_cfgs
from imagecolorizer.utils.load_data import load_image_sample
from imagecolorizer.utils.logger import CustomLogger
import numpy as np

from imagecolorizer.utils.validation import valid_input_data

logger = CustomLogger(name=__name__).get_logger()


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
    config = combine_cfgs(config_file)
    logger.debug(f'Test config: \n {config}')
    image = input_data['instances']

    image = valid_input_data(image, config)

    # Prepare data
    lab = rgb2lab(np.array(image) / 255.)
    l = lab[:, :, 0]
    l_rgb = np.stack((l,) * 3)
    l_rgb = l_rgb.reshape((1, *config.DATA.INPUT_SIZE, 3))

    # Load model
    model = load_model(config=config)

    # Make prediction
    ab = model.predict(l_rgb)
    ab = ab * 128
    color_img = np.zeros((*config.DATA.INPUT_SIZE, 3))

    color_img[:, :, 0] = l
    color_img[:, :, 1:] = ab
    return {"predictions": color_img}


if __name__ == "__main__":
    config = combine_cfgs()
    image = load_image_sample(config)
    input_data = {"instances": image}
    result = make_predictions(input_data=input_data,
                              config_file=None)
    predictions = result.get('predictions')
    expected_prediction_mean_value = np.load(config.TEST.RESULT_PREDICTION)

    assert np.isclose(
        a=np.sum(np.abs(predictions - expected_prediction_mean_value)),
        b=0.0, atol=1e-5, equal_nan=False)
