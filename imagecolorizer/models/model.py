import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     UpSampling2D)


def create_image_colorizer():
    model = Sequential()
    vggmodel = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    for i, layer in enumerate(vggmodel.layers):
        if i < 19:
            model.add(layer)
            if isinstance(layer, Conv2D):
                model.add(BatchNormalization())
    # free the VGG layer
    for layer in model.layers:
        layer.trainable = False

    # Decoder
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    return model
