from keras.applications import VGG16, VGG19, ResNet50, InceptionV3
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from scipy import misc, signal
from matplotlib import pyplot as plt

sobel_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
sobel_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)


def is_conv_layer(weight):
    return len(weight) == 4 and weight[0] > 1


def gradient_x(kernel):
    return np.sum(signal.convolve2d(kernel, sobel_x, mode='valid'))


def gradient_y(kernel):
    return np.sum(signal.convolve2d(kernel, sobel_y, mode='valid'))


def compute_g(model):
    g_sum = 0.0
    for weight in model.get_weights():
        s = weight.shape
        if len(s) < 4 or s[0] != 3:
            continue
        channel = s[2]
        filters = s[3]
        for i in range(channel):
            for j in range(filters):
                g_sum += np.abs(gradient_x(weight[:, :, i, j]))
                g_sum += np.abs(gradient_y(weight[:, :, i, j]))
    return g_sum


def compute_var(model):
    var = 0.0
    for weight in model.get_weights():
        s = weight.shape
        if len(s) < 4:
            continue
        var += np.var(weight)
    return var

config = tf.ConfigProto(
    device_count={'GPU': 0}
)


def print_layers(model):
    for weight in model.get_weights():
        s = weight.shape
        if len(s) == 4 and s[0] > 1: print s, weight.var()

set_session(tf.Session(config=config))
model_vgg_trained = VGG16(include_top=False)
model_vgg_original = VGG16(include_top=False, weights=None)
# model_vgg_19_trained = VGG19(include_top=False)
# model_vgg_19_original = VGG19(include_top=False, weights=None)
model_res_trained = ResNet50(include_top=False)
model_res_original = ResNet50(include_top=False, weights=None)
# model_inc_trained = InceptionV3(include_top=False)
# model_inc_original = InceptionV3(include_top=False, weights=None)
print compute_var(model_vgg_trained), compute_var(model_vgg_original)
# print compute_var(model_vgg_19_trained), compute_var(model_vgg_19_original)
print compute_var(model_res_trained), compute_var(model_res_original)
# print compute_var(model_inc_trained), compute_var(model_inc_original)
print compute_g(model_vgg_trained), compute_g(model_vgg_original)
print compute_g(model_res_trained), compute_g(model_res_original)
for model in [model_vgg_trained, model_vgg_original, model_res_trained, model_res_original]:
    print_layers(model)

for weight in model_res_original.get_weights():
    if is_conv_layer(weight):
        plt.hist(weight.flatten(), bins=100)
        plt.show(block=True)
