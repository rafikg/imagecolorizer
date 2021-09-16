from typing import Optional

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from imagecolorizer.data.data_loader import GrayScaleData, validate_and_load_data
from imagecolorizer.models.model import create_image_colorizer
from imagecolorizer.utils.config_utils import combine_cfgs, create_experiments_dir
from imagecolorizer.utils.logger import CustomLogger
from imagecolorizer.utils.random_seed import set_random_seeds

logger = CustomLogger(name=__name__).get_logger()

set_random_seeds()


def training_pipeline(*, config_file: Optional[str]) -> None:
    # combine config_file with default config
    config = combine_cfgs(cfg_path=config_file)

    logger.debug(f"Training config: \n {config}")
    # create folder where all experiment details will be saved
    create_experiments_dir(config=config)

    # create colorizer_model
    model = create_image_colorizer()

    # load training and validation data
    train_images = validate_and_load_data(config.DATA.TRAIN_PATH)
    valid_images = validate_and_load_data(config.DATA.VALID_PATH)

    # Create train and valid dataset
    train_dataset = GrayScaleData(images_paths=train_images, config=config)
    valid_dataset = GrayScaleData(images_paths=valid_images, config=config)

    # compile the model
    model.compile(optimizer=RMSprop(1e-3), loss="mse")

    # checkpoint
    chpt = tf.keras.callbacks.ModelCheckpoint(
        config.TRAIN.CHECKPOINT,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    # tensorboard
    ts = tf.keras.callbacks.TensorBoard(
        log_dir=config.TRAIN.LOGGING,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    # fitting the model
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config.TRAIN.EPOCHS,
        workers=config.SYSTEM.NUM_WORKERS,
        callbacks=[chpt, ts],
    )

    # Use  saved model format
    tf.saved_model.save(model, config.TRAIN.SAVED_MODELS)
