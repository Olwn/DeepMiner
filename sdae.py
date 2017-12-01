import os
from datetime import datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard, LearningRateScheduler, CSVLogger
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from preprocess import noise
from keras.layers.noise import GaussianNoise
import argparse
import math
from augment_layer import OneDimShift, DependentDense
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import math
from dataset import load_data


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
parser.add_argument("-pre_epochs", type=int, default=50)
parser.add_argument("-train_epochs", type=int, default=200)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-shift_rate", type=float, default=0.1)
parser.add_argument("-gaussian", type=str2bool, default=False)
parser.add_argument("-shift", type=str2bool, default=False)
parser.add_argument("-gpu", type=str, default='1')
parser.add_argument("-mem", type=float, default=0.45)
params = parser.parse_args()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = params.mem
set_session(tf.Session(config=config))

(x_pre_train, y_pre_train), (x_pre_test, y_pre_test) = load_data(params.pretrain_data, params.batch_size)
(x_train, y_train), (x_test, y_test) = load_data(params.test_data, params.batch_size)

dims = [params.units, params.units, params.units]
input1 = Input(shape=(784,), batch_shape=(params.batch_size, 784))
x = input1
x = GaussianNoise(stddev=float(params.gau_std))(x) if params.gaussian else x
x = OneDimShift(shift=params.shift_rate)(x) if params.shift else x
encoded_l1 = Dense(dims[0], activation='sigmoid')
encoded1 = encoded_l1(x)
decoded1 = DependentDense(784, activation='sigmoid', master_layer=encoded_l1)(encoded1)

input2 = Input(shape=(dims[0],), batch_shape=(params.batch_size, dims[0]))
x = input2
x = GaussianNoise(stddev=float(params.gau_std))(x) if params.gaussian else x
# x = OneDimShift(shift=params.shift_rate)(x) if params.shift else x
encoded_l2 = Dense(dims[1], activation='sigmoid')
encoded2 = encoded_l2(x)
decoded2 = DependentDense(dims[0], activation='sigmoid', master_layer=encoded_l2)(encoded2)

input3 = Input(shape=(dims[1],), batch_shape=(params.batch_size, dims[1]))
x = input3
x = GaussianNoise(stddev=float(params.gau_std))(x) if params.gaussian else x
# x = OneDimShift(shift=params.shift_rate)(x) if params.shift else x
encoded_l3 = Dense(dims[2], activation='sigmoid')
encoded3 = encoded_l3(x)
decoded3 = DependentDense(dims[1], activation='sigmoid', master_layer=encoded_l3)(encoded3)

autoencoder1 = Model(input1, decoded1)
encoder1 = Model(input1, encoded1)

autoencoder2 = Model(input2, decoded2)
encoder2 = Model(input2, encoded2)

autoencoder3 = Model(input3, decoded3)
encoder3 = Model(input3, encoded3)

dir_name = "sda/exp" + datetime.now().strftime("%m%d-%H-%M-%S")
if not os.path.exists(dir_name): os.mkdir(dir_name)
with open(os.path.join(dir_name, 'config'), 'w') as wf:
    wf.write("-----" + str(params) + dir_name + "\n")

lr_scheduler = LearningRateScheduler(lambda x: 0.01 if x < params.epochs * 0.5 else 0.001)
# binary_crossentropy
autoencoder1.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder2.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder3.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder1.fit(x_pre_train, x_pre_train,
                epochs=params.pre_epochs, batch_size=params.batch_size, shuffle=True,
                validation_data=(x_pre_test, x_pre_test))
encode1 = encoder1.predict(x_train, batch_size=params.batch_size)

autoencoder2.fit(encode1, encode1,
                 epochs=params.pre_epochs, batch_size=params.batch_size, shuffle=True)

encode2 = encoder2.predict(encode1, batch_size=params.batch_size)

autoencoder3.fit(encode2, encode2,
                 epochs=params.pre_epochs, batch_size=params.batch_size, shuffle=True)

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


def init(layer):
    encoders = [autoencoder1, autoencoder2, autoencoder3]
    def func(shape, dtype=np.float32):
        w = encoders[layer].get_weights()[0]
        assert shape == w.shape
        return w
    return func

input_img = Input(shape=(784,))
hidden1 = Dense(dims[0], kernel_initializer=init(0), activation='sigmoid')(input_img)
hidden2 = Dense(dims[1], kernel_initializer=init(1), activation='sigmoid')(hidden1)
hidden3 = Dense(dims[2], kernel_initializer=init(2), activation='sigmoid')(hidden2)
output = Dense(10, activation='softmax')(hidden3)
classifier = Model(input_img, output)
classifier.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')
csv_logger = CSVLogger(os.path.join(dir_name, 'log.csv'))
board = TensorBoard(write_images=True, log_dir=dir_name)
board.set_model(classifier)
classifier.fit(x_train, y_train, batch_size=params.batch_size, epochs=params.train_epochs,
               validation_data=(x_test, y_test),
               callbacks=[csv_logger, board])
