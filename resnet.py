from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from augment_layer import RandomFlipLayer, RandomShift, MaskLayer, RectifiedPooling2D
import tensorflow as tf

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    kernel_decay = conv_params["kernel_decay"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(kernel_decay))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)

    return f


def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    kernel_decay = conv_params["kernel_decay"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(kernel_decay))

    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False, parser=None):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(x):
        for i in range(repetitions):
            init_strides = (1, 1)
            if parser.my_block or (i == 0 and not is_first_layer):
                init_strides = (2, 2)
            x = block_function(filters=filters, init_strides=init_strides, kernel_decay=parser.kernel_decay,
                               is_first_block_of_first_layer=(is_first_layer and i == 0))(x)
            if parser and parser.mask and parser.mask_more and i != repetitions - 1:
                x = MaskLayer(mask_size=int(K.int_shape(x)[1]*parser.mask_ratio), mask_num=parser.mask_num,
                              mode=parser.mask_mode, drop_rate=parser.drop_rate, params=parser)(x)
        return x

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, kernel_decay=1e-4, is_residual=True):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(kernel_decay))(input)
        else:
            conv1 = conv_bn_relu(filters=filters, kernel_size=(3, 3),
                                 strides=init_strides,
                                 kernel_decay=kernel_decay)(input)

        residual = conv_bn_relu(filters=filters, kernel_size=(3, 3), kernel_decay=kernel_decay)(conv1)
        return _shortcut(input, residual) if is_residual else residual

    return f


def my_block(filters, init_strides=(2, 2), is_first_block_of_first_layer=False):
    def f(input):
        print "my_block is used."
        residual = bn_relu_conv(filters=filters, kernel_size=(5, 5), strides=init_strides)(input)
        return _shortcut(input, residual)
    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, is_residual=True):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                    strides=init_strides)(input)

        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, parser):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        batch_size = parser.batch_size
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")

        # Load function from str if needed.
        # block_fn = _get_block(block_fn)
        block_fn = my_block if parser.my_block else basic_block

        input = Input(shape=input_shape, batch_shape=(batch_size, input_shape[0], input_shape[1], input_shape[2]))
        # input = Input(shape=input_shape, name='input')

        x = conv_bn_relu(filters=parser.filters1, kernel_size=(parser.kernel1, parser.kernel1), strides=(1, 1),
                         kernel_decay=parser.kernel_decay)(input)
        if parser.dataset in ['mit67']:
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        filters = 16 * parser.k
        feature_maps = []
        for i, r in enumerate(repetitions):
            mask_size = int(K.int_shape(x)[1] * parser.mask_ratio)
            x = Dropout(rate=0.5)(x) if parser.dropout else x
            x = RandomShift(shift=0.15)(x) if parser.middle_shift else x
            x = MaskLayer(mask_size=mask_size, mask_num=parser.mask_num,
                          mode=parser.mask_mode, drop_rate=parser.drop_rate, params=parser)(x) if parser.mask else x
            x = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0),
                                parser=parser)(x)
            feature_maps.append(x)

            filters *= 2
        # Last activation
        block = bn_relu(x)

        # Classifier block
        block_shape = K.int_shape(block)
        if parser.global_pool:
            pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(block)
            # pool2 = RectifiedPooling2D(block_shape)(block)
        else:
            block = Conv2D(filters=10, kernel_size=(1, 8))(block)
            block = Conv2D(filters=10, kernel_size=(8, 1))(block)
            pool2 = block
        flatten1 = Flatten()(pool2)
        # flatten1 = Dropout(rate=0.5)(flatten1) if parser.dropout else flatten1
        vision_model = Model(input, flatten1)
        shared_map = Model(input, feature_maps[-1])

        dense_10 = Dense(units=10, kernel_initializer='he_normal', activation='softmax')
        dense_100 = Dense(units=100, kernel_initializer='he_normal', activation='softmax')
        sigmoid_10 = Dense(units=10, kernel_initializer='he_normal', activation='sigmoid')

        models = {
            'cla': Model(input, dense_10(flatten1)),
            # 'cla_10': Model(input_a, dense_10(Flatten()(pool_a_2))),
            # 'cla_100': Model(input_b, dense_100(Flatten()(pool_b_2))),
            # 'tag_10': Model(input_b, sigmoid_10(Flatten()(pool_b_2)))
        }
        return models

    @staticmethod
    def build_resnet(input_shape, num_outputs, parser):
        n = (parser.layers - 2) // 6
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [n, n, n], parser)

    @staticmethod
    def build_resnet_n(input_shape, num_outputs, n, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [n, n, n], parser)

    @staticmethod
    def build_resnet_8(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1], parser)

    @staticmethod
    def build_resnet_14(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2], parser)

    @staticmethod
    def build_resnet_10(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1, 1], parser)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 3, 3], parser)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [5, 5, 5], parser)

    @staticmethod
    def build_resnet_44(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [7, 7, 7], parser)

    @staticmethod
    def build_resnet_56(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [9, 9, 9], parser)

    @staticmethod
    def build_resnet_110(input_shape, num_outputs, parser):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [16, 16, 16], parser)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
