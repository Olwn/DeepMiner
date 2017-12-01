from datetime import datetime
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input as res_preprocess
from keras.datasets import cifar10, cifar100
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from resnet import ResnetBuilder
from keras import backend as K
import argparse
import numpy as np

import matplotlib
from cycler import cycler
from MulticoreTSNE import MulticoreTSNE as TSNE

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot(X, Y, name):
    tsne = TSNE(n_jobs=20)
    X = tsne.fit_transform(X)
    digits = set(Y)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_prop_cycle(
        cycler('color', [colormap(i) for i in np.linspace(0, 0.9, len(digits))]))
    ax = fig.add_subplot(111)
    labels = []
    for d in digits:
        idx = Y == d
        ax.plot(X[idx, 0], X[idx, 1], 'o')
        labels.append(d)
    ax.legend(labels, numpoints=1, fancybox=True)
    fig.savefig(name)


K.set_image_data_format('channels_last')
input_shape = (3, 32, 32)
parser = argparse.ArgumentParser()
parser.add_argument("-flip", type=str2bool, default=False)
parser.add_argument("-shift", type=float, default=0.0)
parser.add_argument("-middle_shift", type=str2bool, default=False)
parser.add_argument("-middle_shift_rate", type=float, default=0.15)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-name", type=str, default="")
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-dataset", type=str, default="cifar10")
parser.add_argument("-layers", type=int, default=8)
parser.add_argument("-gpu", type=str, default="0,1")
parser.add_argument("-crop_scale", type=int, default=1)
parser.add_argument("-mem", type=float, default=0.95)
parser.add_argument("-middle_flip", type=str2bool, default=False)
parser.add_argument("-global_pool", type=str2bool, default=True)
parser.add_argument("-mask", type=str2bool, default=False)
parser.add_argument("-mask_ratio", type=float, default=0.5)
parser.add_argument("-weights", type=str, default="")
parser.add_argument("-begin", type=int, default=0)
parser.add_argument("-patch_flip", type=str2bool, default=False)
parser.add_argument("-patch_option", type=int, default=0)
parser.add_argument("-k", type=int, default=1)
parser.add_argument("-filters1", type=int, default=16)
par = parser.parse_args()

batch_size = par.batch_size
nb_classes = 10 if par.dataset == 'cifar10' else 100
# The data, shuffled and split between train and test sets:
if par.dataset == 'cifar10':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
else:
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train[:batch_size*(X_train.shape[0]/batch_size)]
Y_train = Y_train[:batch_size*(X_train.shape[0]/batch_size)]
X_test = X_test[:batch_size*(X_test.shape[0]/batch_size)]
Y_test = Y_test[:batch_size*(X_test.shape[0]/batch_size)]

# subtract mean and normalize
mean = [125.3, 123.0, 113.9]
std = [63.0,  62.1,  66.7]
mean_image = np.mean(X_train, axis=0)
X_train -= np.reshape(mean, [1, 1, 1, 3])
X_test -= np.reshape(mean, [1, 1, 1, 3])
X_train /= np.reshape(std, [1, 1, 1, 3])
X_test /= np.reshape(std, [1, 1, 1, 3])

# build res-18 network
res_18_model = ResnetBuilder.build_resnet_18(input_shape, nb_classes, par)
if par.weights: res_18_model.load_weights(par.weights, by_name=True)
res_18_extractor = Model(inputs=res_18_model.input, outputs=res_18_model.get_layer('flatten_1').output)

# build res-50-imagenet network
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')
# res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for l in vgg_model.layers: print l.name
vgg_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('global_average_pooling2d_1').output)
# res_extractor = Model(inputs=res_model.input, outputs=res_model.get_layer('avg_pool').output)

model = vgg_extractor
# model = res_18_extractor
features = model.predict(X_test, batch_size=batch_size)
print features.shape
np.save('features.npy', features)
plot(features.astype(np.float64), Y_test.flatten(), 'cifar10_%s.png' % datetime.now().strftime("%m%d-%H-%M-%S"))
