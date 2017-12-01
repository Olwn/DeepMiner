# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
import tensorflow as tf


class RandomShift(Layer):

    def __init__(self, shift=0.0, **kwargs):
        super(RandomShift, self).__init__(**kwargs)
        self.shift = shift
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        print "middle shift layer used"
        input_shape = self.input_spec[0].shape
        shift_pixel = int(input_shape[1] * self.shift)
        padding_shape = [[0, 0], [shift_pixel, shift_pixel], [shift_pixel, shift_pixel], [0, 0]]
        crop = tf.random_crop(tf.pad(x, padding_shape, 'SYMMETRIC'), input_shape)
        return K.in_train_phase(crop, x)

    def get_config(self):
        config = {'shift': self.shift}
        base_config = super(RandomShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
