# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
import tensorflow as tf
from tensorflow.python.framework import dtypes


class RandomFlipLayer(Layer):

    def __init__(self, p=0.5, **kwargs):
        super(RandomFlipLayer, self).__init__(**kwargs)
        self.p = p
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        print "flip layer used"
        has_flip = tf.greater(self.p, tf.random_uniform([1], minval=0, maxval=1))
        flipped = K.switch(has_flip[0], tf.reverse(x, axis=[2]), x)
        return K.in_train_phase(flipped, x)

    def get_config(self):
        config = {'p': self.p}
        base_config = super(RandomFlipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
