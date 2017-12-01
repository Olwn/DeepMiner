from __future__ import division
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, Activation, Flatten,
    MaxPooling2D, BatchNormalization, AveragePooling2D,
    Lambda, InputLayer, Dropout
)
from keras.initializers import RandomNormal
from keras import backend as K
from augment_layer import MaskLayer, MaxAvgPool
import numpy as np
import tensorflow as tf


def smooth_norm(stddev):
    def init(shape, dtype=None):
        print shape
        rnd = K.random_normal(shape=shape, mean=0., stddev=stddev, dtype=dtype)

        rnd_copy = rnd.copy()
        print rnd.shape, rnd_copy.shape
        k_size = shape[1]
        for i in range(1, k_size):
            for j in range(1, k_size):
                i_s = [i - 1, i, i + 1]
                j_s = [j - 1, j, j + 1]
                x = rnd_copy[np.array(i_s), np.array(j_s)]
                rnd[i, j] = np.mean(x, axis=(0, 1))
        return rnd
    return init


class LeNet:
    @staticmethod
    def build(input_shape, classes, weights=None, k=32, params=None):
        # initialize the model
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape, batch_size=params.batch_size))
        # 32 * 32
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same"))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same"))
        if params.bn: model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 16 * 16
        if params.mask: model.add(MaskLayer(mode=params.mask_mode, params=params))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
        if params.bn: model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 8 * 8
        """
        if params.mask: model.add(MaskLayer(mode=params.mask_mode, params=params))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        if params.bn: model.add(BatchNormalization())
        model.add(Activation("relu"))
        """
        # model.add(AveragePooling2D(pool_size=(8, 8)))

        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation(activation='relu'))
        if params.dropout: model.add(Dropout(rate=0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        if weights is not None:
            model.load_weights(weights)

        return model
