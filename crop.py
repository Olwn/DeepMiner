# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
import numpy as np
import tensorflow as tf


class PaddingCrop(Layer):

    def __init__(self, padding=2, flip=False, **kwargs):
        super(PaddingCrop, self).__init__(**kwargs)
        self.padding = padding
        self.flip = flip
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        padding_tensor = tf.pad(x, [[0, 0], [self.padding, self.padding],
                                    [self.padding, self.padding], [0, 0]])
        cropped = tf.random_crop(padding_tensor, input_shape)
        return K.in_train_phase(cropped, x)

    def get_config(self):
        config = {'padding': self.padding, 'flip': self.flip}
        base_config = super(PaddingCrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Cropping2D(Layer):
    '''Cropping layer for 2D input (e.g. picture).
    It crops along spatial dimensions, i.e. width and height.

    # Arguments
        cropping: tuple of tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the 2 cropping dimensions (width, height).
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop)

    # Output shape
        4D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis)

    # Examples

    ```python
        # crop the input image and feature meps
        model = Sequential()
        model.add(Cropping2D(cropping=((2, 2), (4, 4)), input_shape=(3, 28, 28)))
        # now model.output_shape == (None, 3, 24, 20)
        model.add(Convolution2D(64, 3, 3, border_mode='same))
        model.add(Cropping2D(cropping=((2, 2), (2, 2))))
        # now model.output_shape == (None, 64, 20, 16)

    ```

    '''

    def __init__(self, cropping=0, dim_ordering='default', **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.cropping = cropping
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping,
                    input_shape[3] - self.cropping)
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] - self.cropping,
                    input_shape[2] - self.cropping,
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.dim_ordering == 'th':
            return x[:, :, self.cropping:input_shape[2] - self.cropping,
                   self.cropping[1][0]:input_shape[3] - self.cropping]
        elif self.dim_ordering == 'tf':
            has_flip = tf.greater(0.5, tf.random_uniform([1], minval=0, maxval=1))
            x = K.switch(has_flip[0], tf.reverse(x, axis=[2]), x)
            return tf.random_crop(x, [input_shape[0], input_shape[1] - self.cropping,
                                      input_shape[2] - self.cropping, input_shape[3]])
            # return x[:, y_begin:y_begin + input_shape[1] - self.cropping,
            #       x_begin:x_begin + input_shape[2] - self.cropping, :]

    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
