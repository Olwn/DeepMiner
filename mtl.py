"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.
"""
from __future__ import print_function
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger, TensorBoard,
    LearningRateScheduler,
    ModelCheckpoint
)
from keras.backend.tensorflow_backend import set_session
from keras import metrics
import tensorflow as tf
import numpy as np
import resnet
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from os import path as osp
import os
from preprocess import flip, multi_flip, column_mask, mask, flip2
from dataset import mit_scene_67, mix_augmentation, clip_stich_augmentation


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-flip", type=str2bool, default=True)
parser.add_argument("-shift", type=float, default=0.15)
parser.add_argument("-middle_shift", type=str2bool, default=False)
parser.add_argument("-middle_shift_rate", type=float, default=0.15)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-name", type=str, default="")
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-dataset", type=str, default="cifar10")
parser.add_argument("-layers", type=int, default=8)
parser.add_argument("-gpu", type=str, default="0")
parser.add_argument("-crop_scale", type=int, default=1)
parser.add_argument("-mem", type=float, default=0.45)
parser.add_argument("-middle_flip", type=str2bool, default=False)
parser.add_argument("-global_pool", type=str2bool, default=True)
parser.add_argument("-mask", type=str2bool, default=False)
parser.add_argument("-mask_mode", type=str, default='diff')
parser.add_argument("-mask_ratio", type=float, default=0.25)
parser.add_argument("-mask_num", type=int, default=1)
parser.add_argument("-mask_more", type=str2bool, default=False)
parser.add_argument("-drop_rate", type=float, default=0.5)
parser.add_argument("-image_mask", type=str2bool, default=False)
parser.add_argument("-weights", type=str, default="")
parser.add_argument("-begin", type=int, default=2)
parser.add_argument("-patch_flip", type=str2bool, default=False)
parser.add_argument("-patch_option", type=int, default=0)
parser.add_argument("-k", type=int, default=1)
parser.add_argument("-filters1", type=int, default=16)
parser.add_argument("-kernel1", type=int, default=3)
parser.add_argument("-part", type=float, default=1.0)
parser.add_argument("-flip2", type=str2bool, default=False)
parser.add_argument("-meanstd", type=str2bool, default=True)
parser.add_argument("-my_block", type=str2bool, default=False)
parser.add_argument("-smooth", type=str2bool, default=False)
parser.add_argument("-mix_num", type=int, default=0)
parser.add_argument("-kernel_decay", type=float, default=1e-4)
params = parser.parse_args()
print(params)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = params.mem
set_session(tf.Session(config=config))

# make callbacks
# e = params.epochs

def get_lr_scheduler(epochs, factor=1):
    e = epochs
    f = factor
    return LearningRateScheduler(lambda x: 0.1*f if x < e * 0.6 else 0.01*f if x < e * 0.9 else 0.001*f)

dir_name = "e" + datetime.now().strftime("%m%d-%H-%M-%S") + params.dataset + "layers" + \
           str(params.layers) + "e" + str(params.epochs)
if not os.path.exists(dir_name): os.mkdir(dir_name)
board = TensorBoard(log_dir=dir_name, histogram_freq=0, write_graph=False, write_images=False)
csv_logger = CSVLogger(osp.join(dir_name, 'log.csv'), append=True)
checker = ModelCheckpoint(filepath=osp.join(dir_name, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                          save_weights_only=False, period=20)
with open(osp.join(dir_name, 'config'), 'w') as wf:
    wf.write("-----" + str(params) + dir_name + "\n")

# prepare data
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()

# use a part of training set
n = x_train_10.shape[0]
x_train_10 = x_train_10[:int(n * params.part)]
y_train_10 = y_train_10[:int(n * params.part)]
x_train_100 = x_train_100[:int(n * params.part)]
y_train_100 = y_train_100[:int(n * params.part)]
input_shape = (32, 32, 3)

# Convert class vectors to binary class matrices.
y_train_10 = np_utils.to_categorical(y_train_10, 10)
y_test_10 = np_utils.to_categorical(y_test_10, 10)
y_train_100 = np_utils.to_categorical(y_train_100, 100)
y_test_100 = np_utils.to_categorical(y_test_100, 100)

x_train_10 = x_train_10.astype('float32')
x_test_10 = x_test_10.astype('float32')
x_train_100 = x_train_100.astype('float32')
x_test_100 = x_test_100.astype('float32')
x_train_10_2, y_train_10_2 = mix_augmentation(x_train_10, y_train_10, params.mix_num)

batch_size = params.batch_size

if params.layers % 6 != 2:
    raise argparse.ArgumentError("layers must be in [8, 18, 34, 44, 56, 110]")
# cla_model, tag_model = resnet.ResnetBuilder.build_resnet(input_shape, nb_classes, params)
models = resnet.ResnetBuilder.build_resnet(input_shape, 0, params)
model_cla_10 = models['cla_10']
model_cla_100 = models['cla_100']
model_tag_10 = models['tag_10']
model_cla_10.compile(loss=['categorical_crossentropy'], optimizer='sgd', metrics=['accuracy'])
model_cla_100.compile(loss=['categorical_crossentropy'], optimizer='sgd', metrics=['accuracy'])
model_tag_10.compile(loss=['binary_crossentropy'], optimizer='sgd', metrics=['accuracy'])

datagen_train = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=params.shift,
    height_shift_range=params.shift,
    horizontal_flip=params.flip,
    vertical_flip=False,
    preprocessing_function=None
)
datagen_test = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False
)
# if params.weights != "": model.load_weights(params.weights, by_name=True)
for i in range(1, params.epochs, 2):
    model_cla_10.fit_generator(
        datagen_train.flow(x_train_10, y_train_10, batch_size=batch_size, seed=i),
        steps_per_epoch=x_train_10.shape[0] / batch_size,
        validation_data=datagen_test.flow(x_test_10, y_test_10, batch_size=batch_size),
        validation_steps=50,
        epochs=i,
        max_q_size=250,
        workers=5,
        initial_epoch=i - 1,
        callbacks=[get_lr_scheduler(params.epochs, 1), csv_logger])
    model_tag_10.fit_generator(
        datagen_train.flow(x_train_10_2, y_train_10_2, batch_size=batch_size, seed=i),
        steps_per_epoch=x_train_10.shape[0] / batch_size,
        # validation_data=datagen_test.flow(x_test_10, y_test_10, batch_size=batch_size),
        # validation_steps=50,
        epochs=i,
        max_q_size=250,
        workers=5,
        initial_epoch=i - 1,
        callbacks=[get_lr_scheduler(params.epochs, 0.5), csv_logger])
    """
    model_cla_100.fit_generator(
        datagen_train.flow(x_train_100, y_train_100, batch_size=batch_size, seed=i),
        steps_per_epoch=x_train_100.shape[0] / batch_size,
        validation_data=datagen_test.flow(x_test_100, y_test_100, batch_size=batch_size),
        validation_steps=50,
        epochs=i + 1,
        max_q_size=250,
        workers=5,
        initial_epoch=i,
        callbacks=[lr_scheduler, csv_logger])
    """

losses_10 = model_cla_10.evaluate(x_test_10, y_test_10, batch_size=params.batch_size)
losses_100 = model_cla_100.evaluate(x_test_100, y_test_100, batch_size=params.batch_size)
print(losses_10, losses_100)
with open(osp.join(dir_name, 'config'), 'a') as wf:
    wf.write(str(losses_10) + '\n')
    wf.write(str(losses_100) + '\n')
"""
output = model.predict(x_test, batch_size=batch_size)
labels = output.argmax(axis=1)
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i, label in enumerate(labels):
    if label != y_test_label[i]:
        plt.imshow(X_test_image[i])
        file_name = "%s_%d_%s.jpg" % (names[y_test_label[i]], i, names[label])
        plt.savefig(osp.join(dir_name, file_name))
"""