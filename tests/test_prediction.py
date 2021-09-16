import numpy as np

from imagecolorizer.models.inference import make_predictions
from imagecolorizer.utils.config_utils import combine_cfgs
from imagecolorizer.utils.random_seed import set_random_seeds

config = combine_cfgs()
set_random_seeds()


def test_make_prediction(sample_input_data):
    # Given
    expected_prediction_mean_value = np.load(config.TEST.RESULT_PREDICTION)
    # When
    result = make_predictions(
        input_data={"instances": sample_input_data}, config_file=None
    )

    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0, 0, 0], np.float64)
    assert predictions.ndim == 3
    assert np.isclose(
        a=np.sum(np.abs(predictions - expected_prediction_mean_value)),
        b=0.0,
        atol=1e-5,
        equal_nan=False,
    )
