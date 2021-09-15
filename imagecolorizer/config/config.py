import os
from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

_C.ROOT_DIR = str((Path(__file__)).parent.parent)

_C.SYSTEM = CN()

# Number of GPU to use
_C.SYSTEM.NUM_GPUS = 2
# Number of workers to use
_C.SYSTEM.NUM_WORKERS = 4

_C.DATA = CN()

# Train data path
_C.DATA.TRAIN_PATH = os.path.join(_C.ROOT_DIR, "datasets", "train")

# Valid data path
_C.DATA.VALID_PATH = os.path.join(_C.ROOT_DIR, "datasets", "valid")
# input image size
_C.DATA.INPUT_SIZE = (224, 224)
_C.DATA.SHUFFLE = True

_C.TRAIN = CN()

# Number of epochs
_C.TRAIN.EPOCHS = 100
# Number of batch_size
_C.TRAIN.BATCH_SIZE = 32
# Base learning rate
_C.TRAIN.BASE_LR = 0.001
# Experiment number
_C.TRAIN.EXP_NUM = "exp0001"

# output dir
_C.TRAIN.OUTPUT_DIR = os.path.join(_C.ROOT_DIR, "experiments", _C.TRAIN.EXP_NUM)
# Checkpoint path
_C.TRAIN.CHECKPOINT = os.path.join(_C.TRAIN.OUTPUT_DIR, "checkpoint")
# Logging
_C.TRAIN.LOGGING = os.path.join(_C.TRAIN.OUTPUT_DIR, "logs")
# Saved models
_C.TRAIN.SAVED_MODELS = os.path.join(_C.TRAIN.OUTPUT_DIR, "saved_models")

# Test data
_C.TEST = CN()
_C.TEST.IMAGE_SAMPLE_PATH = os.path.join(
    _C.ROOT_DIR, "../" "tests", "sample_data", "sample000.jpg"
)
_C.TEST.RESULT_PREDICTION = os.path.join(
    _C.ROOT_DIR, "../", "tests", "sample_data", "test_predictions.npy"
)
_C.TEST.TRAINED_MODEL = os.path.join(_C.ROOT_DIR, "trained_models", "best_weights.h5")


def get_cfg_defaults():
    """
    :get_cfg_defaults: Get a yacs CfgNode object with default values
    :return:
        clone the default object
    """
    return _C.clone()
