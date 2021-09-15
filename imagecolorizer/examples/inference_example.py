import numpy as np

from imagecolorizer.models.inference import make_predictions
from imagecolorizer.utils.config_utils import combine_cfgs
from imagecolorizer.utils.load_data import load_image_sample

config = combine_cfgs()
image = load_image_sample(config=config)

input_data = {"instances": image}

result = make_predictions(input_data=input_data, config_file=None)
predictions = result.get("predictions")
expected_prediction_mean_value = np.load(config.TEST.RESULT_PREDICTION)

assert np.isclose(
    a=np.sum(np.abs(predictions - expected_prediction_mean_value)),
    b=0.0,
    atol=1e-5,
    equal_nan=False,
)
