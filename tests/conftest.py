import pytest

from imagecolorizer.utils.config_utils import combine_cfgs
from imagecolorizer.utils.load_data import load_image_sample

config = combine_cfgs(cfg_path=None)


@pytest.fixture()
def sample_input_data():

    return load_image_sample(config=config)
