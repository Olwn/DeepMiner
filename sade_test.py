import os
from datetime import datetime
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard, LearningRateScheduler, CSVLogger
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from preprocess import noise
from keras import backend as K
from keras.layers.noise import GaussianNoise
import argparse
import math
from augment_layer import OneDimShift
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import math
from dataset import load_mnist

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-units", type=int, default=1000)
parser.add_argument("-pretrain_data", type=str, default='basic')
parser.add_argument("-test_data", type=str, default='rndback')
parser.add_argument("-gau_std", type=float, default=0.5)
parser.add_argument("-epochs", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-shift_rate", type=float, default=0.1)
parser.add_argument("-gaussian", type=str2bool, default=False)
parser.add_argument("-shift", type=str2bool, default=False)
parser.add_argument("-gpu", type=str, default='1')
parser.add_argument("-mem", type=float, default=0.45)
params = parser.parse_args()


def load_data(mode):
    (x_train, y_train), (x_test, y_test) = load_mnist(mode)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    batch_size = params.batch_size
    cut = batch_size * (x_train.shape[0] / batch_size)
    x_train = x_train[:cut]
    y_train = y_train[:cut]
    cut = batch_size * (x_test.shape[0] / batch_size)
    x_test = x_test[:cut]
    y_test = y_test[:cut]
    return (x_train, y_train), (x_test, y_test)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = params.mem
set_session(tf.Session(config=config))

(x_pre_train, y_pre_train), (x_pre_test, y_pre_test) = load_data(params.pretrain_data)
(x_train, y_train), (x_test, y_test) = load_data(params.test_data)

dims = [params.units, params.units, params.units]
input1 = Input(shape=(784,), batch_shape=(params.batch_size, 784))
x = input1
x = GaussianNoise(stddev=float(params.gau_std))(x)
encoded1 = Dense(dims[0], activation='sigmoid')(x)
decoded1 = Dense(784, activation='sigmoid')(encoded1)

input2 = Input(shape=(784,), batch_shape=(params.batch_size, 784))
x = input2
x = GaussianNoise(stddev=float(params.gau_std))(x)
x = OneDimShift(shift=params.shift_rate)(x)
encoded2 = Dense(dims[0], activation='sigmoid')(x)
decoded2 = Dense(784, activation='sigmoid')(encoded2)

input3 = Input(shape=(dims[1],), batch_shape=(params.batch_size, dims[1]))
x = input3
x = GaussianNoise(stddev=float(params.gau_std))(x) if params.gaussian else x
# x = OneDimShift(shift=params.shift_rate)(x) if params.shift else x
encoded3 = Dense(dims[2], activation='sigmoid')(x)
decoded3 = Dense(dims[1], activation='sigmoid')(encoded3)

autoencoder1 = Model(input1, decoded1)
encoder1 = Model(input1, encoded1)

autoencoder2 = Model(input2, decoded2)
encoder2 = Model(input2, encoded2)

# autoencoder3 = Model(input3, decoded3)
# encoder3 = Model(input3, encoded3)

dir_name = "sda/exp" + datetime.now().strftime("%m%d-%H-%M-%S")
if not os.path.exists(dir_name): os.mkdir(dir_name)
with open(os.path.join(dir_name, 'config'), 'w') as wf:
    wf.write("-----" + str(params) + dir_name + "\n")

# binary_crossentropy
autoencoder1.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder2.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder3.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder1.fit(x_pre_train, x_pre_train, epochs=params.epochs, batch_size=params.batch_size, shuffle=True)
autoencoder2.fit(x_pre_train, x_pre_train, epochs=params.epochs, batch_size=params.batch_size, shuffle=True)
# autoencoder3.fit(encode2, encode2, epochs=params.epochs, batch_size=params.batch_size, shuffle=True)

"""
filters = autoencoder1.get_weights()[0]
plt.axis('off')
r = min(int(math.sqrt(params.units)), 10)
f, axarr = plt.subplots(r, r)
for i in range(r):
    for j in range(r):
        idx = i*r+j
        img = filters[:, idx].reshape((28, 28))
        axarr[i, j].imshow(img, cmap=plt.cm.get_cmap('gray'))
        axarr[i, j].axis('off')
f.savefig(os.path.join(dir_name, 'filters.jpg'))
"""


def init(layer):
    encoders = [autoencoder1, autoencoder2]

    def func(shape, dtype=np.float32):
        w = encoders[layer].get_weights()[0]
        assert shape == w.shape
        return w
    return func

input_img = Input(shape=(784,))
hidden1 = Dense(dims[0], kernel_initializer=init(0), activation='sigmoid')(input_img)
hidden2 = Dense(dims[0], kernel_initializer=init(1), activation='sigmoid')(input_img)
hidden3 = concatenate([hidden1, hidden2])
print K.shape(hidden1), K.shape(hidden2), K.shape(hidden3)
output = Dense(10, activation='softmax')(hidden3)
classifier = Model(input_img, output)
classifier.compile(optimizer=Adadelta(), metrics=['accuracy'], loss='categorical_crossentropy')
csv_logger = CSVLogger(os.path.join(dir_name, 'log.csv'))
board = TensorBoard(write_images=True, log_dir=dir_name)
board.set_model(classifier)
epochs = 200
lr_scheduler = LearningRateScheduler(lambda e: 1e-2 if e < 100 else 1e-3 if e < 150 else 1e-4)
classifier.fit(x_train, y_train, batch_size=params.batch_size, epochs=200,
               validation_data=(x_test, y_test),
               callbacks=[csv_logger, board])
