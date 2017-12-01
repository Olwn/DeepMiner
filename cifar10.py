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
from keras import backend as K
import tensorflow as tf
import numpy as np
import resnet
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from os import path as osp
import os
from preprocess import flip, multi_flip, column_mask, mask, flip2, contrast, bright
from dataset import mit_scene_67, mix_augmentation, clip_stich_augmentation
from lenet import LeNet
import string
from callback import PrintWeights

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
parser.add_argument("-image_contrast", type=str2bool, default=False)
parser.add_argument("-image_bright", type=str2bool, default=False)
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
parser.add_argument("-network", type=str, default='lenet')
parser.add_argument("-dropout", type=str2bool, default=False)
parser.add_argument("-topk", type=int, default=32)
parser.add_argument("-bn_first", type=str2bool, default=True)
parser.add_argument("-bright", type=float, default=0.5)
parser.add_argument("-contrast", type=float, default=0.75)
parser.add_argument("-bn", type=str2bool, default=True)
params = parser.parse_args()
print(params)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = params.mem
set_session(tf.Session(config=config))

# make callbacks
e = params.epochs
if e == 200:
    lr_scheduler = LearningRateScheduler(lambda x: 0.1 if x < 120 else 0.01 if x < 160 else 0.001)
elif e == 100:
    lr_scheduler = LearningRateScheduler(lambda x: 0.05 if x < 60 else 0.03 if x < 80 else 0.005)
elif e == 400:
    lr_scheduler = LearningRateScheduler(lambda x: 0.1 if x < 300 else 0.01 if x < 350 else 0.001)
else:
    lr_scheduler = LearningRateScheduler(lambda x: 0.1 if x < e * 0.6 else 0.01 if x < e * 0.9 else 0.003)
dir_name = "e" + datetime.now().strftime("%m%d-%H-%M-%S") + params.dataset + "layers" + \
           str(params.layers) + "e" + str(params.epochs) + "_ing"

board = TensorBoard(log_dir=dir_name, histogram_freq=0, write_graph=False, write_images=False)
csv_logger = CSVLogger(osp.join(dir_name, 'log.csv'))
checker = ModelCheckpoint(filepath=osp.join(dir_name, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"), period=20)

# prepare data
if params.dataset == 'cifar10':
    data = cifar10.load_data()
elif params.dataset == 'cifar100':
    data = cifar100.load_data()
elif params.dataset == 'mit67':
    data = mit_scene_67(100)
else:
    raise argparse.ArgumentError("invalid dataset setting")
(x_train, y_train), (x_test, y_test) = data
# use a part of training set
n = x_train.shape[0]
x_train = x_train[:int(n * params.part)]
y_train = y_train[:int(n * params.part)]
nb_classes = y_train.max() + 1
input_shape = (32, 32, 3)
X_test_image = x_test
y_test_label = y_test
# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
x_train_2, y_train_2 = mix_augmentation(x_train, y_train, params.mix_num)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

batch_size = params.batch_size
x_train = x_train[:batch_size * (x_train.shape[0] / batch_size)]
y_train = y_train[:batch_size * (x_train.shape[0] / batch_size)]
x_test = x_test[:batch_size * (x_test.shape[0] / batch_size)]
y_test = y_test[:batch_size * (x_test.shape[0] / batch_size)]
x_train_2 = x_train_2[:batch_size * (x_train_2.shape[0] / batch_size)]
y_train_2 = y_train_2[:batch_size * (x_train_2.shape[0] / batch_size)]

# subtract mean and normalize
if params.meanstd:
    mean = [125.3, 123.0, 113.9]
    std = [63.0,  62.1,  66.7]
    mean_image = np.mean(x_train, axis=0)
    x_train -= np.reshape(mean, [1, 1, 1, 3])
    x_test -= np.reshape(mean, [1, 1, 1, 3])
    x_train /= np.reshape(std, [1, 1, 1, 3])
    x_test /= np.reshape(std, [1, 1, 1, 3])


if params.network == 'resnet':
    if params.layers % 6 != 2:
        raise argparse.ArgumentError("layers-2 must be dividable by six")
    models = resnet.ResnetBuilder.build_resnet(input_shape, nb_classes, params)
    model = models['cla']
else:
    model = LeNet.build(input_shape, nb_classes, k=params.topk, params=params)
model.compile(loss=['categorical_crossentropy'], optimizer=SGD(momentum=0.9, decay=1e-4), metrics=['accuracy'])

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
    preprocessing_function=contrast if params.image_contrast else bright if params.image_bright else None
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
if params.weights != "": model.load_weights(params.weights, by_name=False)
if not os.path.exists(dir_name): os.mkdir(dir_name)
with open(osp.join(dir_name, 'config'), 'w') as wf:
    wf.write("-----" + str(params) + dir_name)

callbacks = [lr_scheduler, csv_logger, board, checker]
if params.smooth:
    callbacks.append(PrintWeights(model=model))

model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] / batch_size,
                    validation_data=datagen_test.flow(x_test, y_test, batch_size=batch_size),
                    validation_steps=78,
                    epochs=params.epochs + 2,
                    max_q_size=2500,
                    workers=1,
                    initial_epoch=params.begin,
                    callbacks=callbacks)

losses = model.evaluate(x_test, y_test, batch_size=params.batch_size)
print(losses)
with open(osp.join(dir_name, 'config'), 'a') as wf:
    wf.write(str(losses) + '\n')
os.rename(dir_name, dir_name.replace('_ing', '_completed'))


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