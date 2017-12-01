from __future__ import division

from keras.datasets import cifar10
from keras.layers import (
    Input, Flatten, Dense
)
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from keras.initializers import he_normal
from resnet import bn_relu_conv, bn_relu
import cv2
import numpy as np
from matplotlib import pyplot as plt

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def my_init(shape, dtype=np.float32):
    """
    initialize using gabor filter
    :return: 
    """
    '''
        lamda: size[0], size[0] / 2
        sigma: 0.56 * lamda
        theta: 0 ~ 360
        psi: -180 ~ 180
    '''
    weights = np.zeros(shape, dtype=dtype)
    var = 2.0 / (shape[0] * shape[1] * shape[2])
    kernel_size = shape[0]
    channels = shape[2]
    filters = shape[3]
    for i in range(filters):
        for j in range(channels):
            theta = np.pi * i / filters
            psi = np.pi * (-180 + 360 * j / channels)
            weight = cv2.getGaborKernel((kernel_size, kernel_size), sigma=1,
                                                     theta=theta, lambd=kernel_size*10, gamma=1, psi=0)
            weight -= weight.mean()
            weight /= np.sqrt(np.var(weight) / var)
            weights[:, :, j, i] = weight
    origin_var = np.var(weights)
    scale_factor = np.sqrt(origin_var / var)
    weights /= scale_factor
    print weights.var(), weights.mean()
    # plt.hist(weights.flatten(), bins=100)
    # plt.show()
    return weights


def get_alex_model(input_shape=(32, 32, 3), classes=10):
    img_input = Input(input_shape)
    he_init = he_normal()
    x = bn_relu_conv(kernel_size=(3, 3), filters=64, kernel_initializer=my_init, strides=(1, 1))(img_input)
    x = bn_relu_conv(kernel_size=(3, 3), filters=64, kernel_initializer=my_init, strides=(2, 2))(x)
    x = bn_relu_conv(kernel_size=(3, 3), filters=128, kernel_initializer=my_init, strides=(1, 1))(x)
    x = bn_relu_conv(kernel_size=(3, 3), filters=128, kernel_initializer=my_init, strides=(2, 2))(x)
    x = bn_relu_conv(kernel_size=(3, 3), filters=256, kernel_initializer=my_init, strides=(1, 1))(x)
    x = bn_relu(x)
    x = Flatten()(x)
    dense = Dense(units=classes, kernel_initializer="he_normal", activation="softmax")(x)
    model = Model(inputs=img_input, outputs=dense)
    return model

if __name__ == '__main__':
    nb_classes = 10
    input_shape = (32, 32, 3)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    model = get_alex_model(input_shape, classes=nb_classes)
    model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    for weight in model.get_weights():
        print weight.var()
    model.fit(X_train, Y_train, epochs=100, batch_size=128)
