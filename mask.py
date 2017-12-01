# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
import tensorflow as tf
from tensorflow.python.framework import dtypes


class MaskLayer(Layer):

    def __init__(self, mask_size, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self.mask_size = mask_size
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        print "mask layer used"
        input_shape = self.input_spec[0].shape
        h = input_shape[1]
        w = input_shape[2]
        batch = input_shape[0]
        channel = input_shape[3]
        r = h - self.mask_size
        tx = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
        ty = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
        ones = tf.ones([batch, self.mask_size, self.mask_size, channel])
        mask_one = tf.pad(ones, [[0, 0], [tx, r - tx], [ty, r - ty], [0, 0]])
        mask_zero = tf.subtract(1.0, mask_one)
        return K.in_train_phase(tf.multiply(x, mask_zero), x)

    def get_config(self):
        config = {'mask_size': self.mask_size}
        base_config = super(MaskLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
