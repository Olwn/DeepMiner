# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers import AveragePooling2D, interfaces
import tensorflow as tf
import math
import numpy as np

from keras.layers import Dense
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

    def call(self, x, mask=None, training=None):
        print "flip layer used"
        print training
        has_flip = tf.greater(self.p, tf.random_uniform([1], minval=0, maxval=1))
        flipped = K.switch(has_flip[0], tf.reverse(x, axis=[2]), x)
        return K.in_train_phase(flipped, x, training=training)

    def get_config(self):
        config = {'p': self.p}
        base_config = super(RandomFlipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OneDimShift(Layer):

    def __init__(self, shift=0.0, **kwargs):
        super(OneDimShift, self).__init__(**kwargs)
        self.shift = shift
        self.input_spec = [InputSpec(ndim=2)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, inputs, **kwargs):
        print "shift used"
        def shift():
            shape = self.input_spec[0].shape
            images = tf.reshape(inputs, (shape[0], 28, 28, 1))
            p = int(self.shift * 28)
            padding = tf.pad(images, [[0, 0], [p, p], [p, p], [0, 0]])
            crop = tf.random_crop(padding, size=(shape[0], 28, 28, 1))
            return tf.reshape(crop, (shape[0], 784))

        return K.in_train_phase(shift, inputs)


class RandomShift(Layer):

    def __init__(self, shift=0.0, **kwargs):
        super(RandomShift, self).__init__(**kwargs)
        self.shift = shift
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None, training=None):

        def shift():
            input_shape = self.input_spec[0].shape
            shift_pixel = int(input_shape[1] * self.shift)
            print "shift %d pixels" % shift_pixel

            padding_shape = [[shift_pixel, shift_pixel], [shift_pixel, shift_pixel], [0, 0]]
            slices = tf.unstack(x)
            for i, slice_i in enumerate(slices):
                has_shift = tf.greater(0.1, tf.random_uniform([1], minval=0, maxval=1))
                shifted = tf.random_crop(tf.pad(slice_i, padding_shape, 'REFLECT'), input_shape[1:])
                slices[i] = K.switch(has_shift[0], shifted, slices[i])
            return tf.stack(slices)

            # padding_shape = [[0, 0], [shift_pixel, shift_pixel], [shift_pixel, shift_pixel], [0, 0]]
            # return tf.random_crop(tf.pad(x, padding_shape, 'REFLECT'), input_shape)
        return K.in_train_phase(shift, x, training=training)

    def get_config(self):
        config = {'shift': self.shift}
        base_config = super(RandomShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskNoise(Layer):

    def __init__(self, rate, **kwargs):
        super(MaskNoise, self).__init__(**kwargs)
        self.rate = rate
        self.input_spec = [InputSpec(ndim=2)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, inputs, **kwargs):

        def noise():
            shape = self.input_spec[0].shape
            mask = tf.multinomial(tf.log([[self.rate, 1. - self.rate]]), np.prod(shape))
            return inputs * tf.reshape(tf.cast(mask, dtype=tf.float32), shape)

        return K.in_train_phase(noise, inputs)


class MaskLayer(Layer):

    def __init__(self, mask_size=0, mask_num=1, mode='same', drop_rate=0.5, params=None, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self.mask_size = mask_size
        self.mask_num = mask_num
        self.mode = mode
        self.drop_rate = drop_rate
        self.params = params
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None, training=None):
        shape = self.input_spec[0].shape
        print("mask: %d" % self.mask_size)
        def masked_diff():
            h = shape[1]
            channel = shape[3]
            slices = tf.unstack(x)
            for i, slice_i in enumerate(slices):
                r = h - self.mask_size
                for j in range(self.mask_num):
                    tx = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
                    ty = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
                    ones = tf.ones([self.mask_size, self.mask_size, channel])
                    mask_one = tf.pad(ones, [[tx, r - tx], [ty, r - ty], [0, 0]])
                    mask_zero = tf.subtract(1.0, mask_one)
                    slices[i] = tf.multiply(slices[i], mask_zero)
            out = tf.stack(slices)
            # out = tf.div(slices, 1 - math.pow(self.mask_size / h, 2.))
            return out

        def masked_separate():
            h = shape[1]
            r = h - self.mask_size
            is_row = tf.greater(0.5, tf.random_uniform([1], minval=0, maxval=1))[0]
            t_m = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
            t_1 = tf.random_uniform([1], minval=0, maxval=h-1, dtype=dtypes.int32)[0]
            row_ones = tf.ones([1, 1, self.mask_size, 1])
            col_ones = tf.ones([1, self.mask_size, 1, 1])
            row_padding = tf.pad(row_ones, [[0, 0], [t_1, h-1-t_1], [t_m, r - t_m], [0, 0]])
            col_padding = tf.pad(col_ones, [[0, 0], [t_m, r - t_m], [t_1, h - 1 - t_1], [0, 0]])
            # out = tf.stack(tf.multiply(x, 1.0 - K.switch(is_row, row_padding, col_padding)))
            out = x * (1.0 - row_padding) * (1.0 - col_padding)
            # scaling
            # out = tf.div(out, 1 - math.pow(self.mask_size / h, 2.))
            return out

        def masked_same():
            h = shape[1]
            r = h - self.mask_size
            tx = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
            ty = tf.random_uniform([1], minval=0, maxval=r, dtype=dtypes.int32)[0]
            ones = tf.ones([1, self.mask_size, self.mask_size, 1])
            padding = tf.pad(ones, [[0, 0], [ty, r - ty], [tx, r - tx], [0, 0]])
            out = tf.multiply(x, 1.0 - padding)
            # scaling
            # out = tf.div(out, 1 - math.pow(self.mask_size / h, 2.))
            return out

        def masked_binomial():
            if self.drop_rate < 0.0001:
                return x
            t = shape
            random_mask = tf.multinomial(tf.log([[10.*self.drop_rate, 10.*(1.-self.drop_rate)]]), 128*t[1]*t[2]*t[3])
            random_mask = tf.cast(random_mask, dtype=dtypes.float32)
            return tf.div(x, 1. - self.drop_rate) * tf.reshape(random_mask, (128, t[1], t[2], t[3]))

        def masked_channel():
            channels = shape[3]
            channel_mask = tf.multinomial(tf.log([[self.drop_rate, 1.-self.drop_rate]]), channels)
            channel_mask = tf.cast(channel_mask, dtype=dtypes.float32)
            out = tf.div(x, 1. - self.drop_rate) * tf.reshape(channel_mask, [1, 1, 1, channels])
            return out

        def masked_gaussian():
            mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
            noise = tf.truncated_normal(shape, 0, tf.sqrt(var))
            return x + noise

        def masked_zero():
            t = shape
            zeros = tf.multinomial(tf.log([[self.drop_rate, 1.-self.drop_rate]]), t[0] * t[1] * t[2] * t[3])
            return tf.div(x, 1. - self.drop_rate) * tf.reshape(tf.cast(zeros, dtype=dtypes.float32), shape)

        def random_contrast():
            slices = tf.unstack(x, axis=3)
            for i, slice_i in enumerate(slices):
                factor1 = tf.random_uniform(shape=[1], minval=0.75, maxval=1.25)
                # factor2 = tf.random_uniform(shape=[1], minval=0.5, maxval=1)
                m = tf.reduce_mean(slice_i)
                contrasted1 = (slice_i - m) * factor1 + m
                # contrasted2 = (slice_i - m) * factor2 + m
                # has_aug = tf.greater(0.5, tf.random_uniform([1], minval=0, maxval=1))
                # slices[i] = K.switch(has_aug[0], contrasted1, contrasted2)
                slices[i] = (slice_i - m) * factor1 + m
            # return tf.stack(slices, axis=3)

            # 图像和通道，只调整一个？
            m = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            f = tf.random_uniform(shape=[shape[0], 1, 1, shape[-1]],
                                  minval=self.params.contrast,
                                  maxval=2.0 - self.params.contrast)
            return (x - m) * f + m
            # return tf.image.adjust_contrast(x, contrast_factor=0.9)
            # return tf.image.random_contrast(x, lower=0.9, upper=1.1)

        def random_brightness():
            # 图像和通道，只调整一个？
            m, v = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
            f = tf.random_uniform(shape=[shape[0], 1, 1, shape[-1]],
                                  minval=-1. * self.params.bright,
                                  maxval=self.params.bright)
            return x + tf.sqrt(v) * f


        output = {
            'diff': masked_diff,
            'same': masked_same,
            'zero': masked_zero,
            'channel': masked_channel,
            'gaussian': masked_gaussian,
            'binomial': masked_binomial,
            'contrast': random_contrast,
            'separate': masked_separate,
            'brightness': random_brightness
        }
        return K.in_train_phase(output[self.mode], x, training=training)

    def get_config(self):
        config = {'mask_size': self.mask_size, 'mode': self.mode, 'mask_num': self.mask_num}
        base_config = super(MaskLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxAvgPool(Layer):
    def __init__(self, k, **kwargs):
        super(MaxAvgPool, self).__init__(**kwargs)
        self.k = k
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        print input_shape
        return input_shape[0], input_shape[3]

    def call(self, inputs, **kwargs):
        shape = self.input_spec[0].shape
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        x = tf.reshape(x, shape=(-1, shape[3], shape[1] * shape[2]))
        vals, idxs = tf.nn.top_k(x, k=self.k, sorted=False)
        return tf.reduce_mean(vals, axis=2)


class RectifiedPooling2D(Layer):
    def __init__(self, input_shape, **kwargs):
        super(RectifiedPooling2D, self).__init__(**kwargs)
        self.shape = input_shape
        self.input_spec = [InputSpec(ndim=4)]

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, 1, input_shape[3]

    def call(self, inputs, **kwargs):
        h = self.shape[1]
        pixels = h * h
        pooled = tf.nn.avg_pool(inputs, (1, h, h, 1), strides=(1, 1, 1, 1), padding='VALID')
        nonzero_count = tf.count_nonzero(inputs, axis=[1, 2], keep_dims=False, dtype=dtypes.float32)
        factors = tf.divide(nonzero_count, float(pixels))
        print self.shape
        batches = tf.unstack(factors, num=128, axis=0)
        for i, batch in enumerate(batches):
            channels = tf.unstack(batch)
            for j, f in enumerate(channels):
                channels[j] = K.switch(f < 1e-6, 1., f)
            batches[i] = tf.stack(channels)
        factors = tf.parallel_stack(batches)

        return tf.divide(pooled, tf.reshape(factors, (128, 1, 1, self.shape[3])))


class DependentDense(Dense):
    def __init__(self, units,
                 master_layer,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.master_layer = master_layer
        super(DependentDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = K.transpose(self.master_layer.kernel)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
