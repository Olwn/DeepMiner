from __future__ import division
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.datasets import mnist, cifar10
from os import path as osp
from keras import backend as K

from keras.utils import np_utils
from scipy.misc import imresize
import os
import numpy as np
import pickle
import h5py


def save_cifar_to_hdf5():
    K.set_image_data_format('channels_first')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print x_train.shape
    h5f = h5py.File('data4torch.h5', 'w')
    h5f.create_dataset(name='data', data=x_train, shape=(50000, 3, 32, 32))
    h5f.create_dataset(name='labels', data=y_train.flatten(), shape=(50000,))
    h5f.close()


def load(path, min_pixel=200):
    img = load_img(path)
    mat = img_to_array(img)
    if mat.shape[0] < mat.shape[1]:
        new_shape = (200, int(mat.shape[1] / mat.shape[0] * min_pixel))
    else:
        new_shape = (int(mat.shape[0] / mat.shape[1] * min_pixel), 200)
    return imresize(mat, new_shape)


def mit_scene_67(pixel=200):
    dir = '/home/x/Downloads/mit67'
    x_train_file = osp.join(dir, 'x_train_%d.npy' % pixel)
    y_train_file = osp.join(dir, 'y_train_%d.npy' % pixel)
    x_test_file = osp.join(dir, 'x_test_%d.npy' % pixel)
    y_test_file = osp.join(dir, 'y_test_%d.npy' % pixel)
    if os.path.exists(x_train_file):
        return (np.load(x_train_file), np.load(y_train_file)),\
               (np.load(x_test_file), np.load(y_test_file))
    names = []
    train_list_file = osp.join(dir, 'TrainImages.txt')
    test_list_file = osp.join(dir, 'TestImages.txt')
    crop_size = 200
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for line in open(train_list_file, 'r').readlines():
        img = load(osp.join(dir, line.rstrip('\n')), crop_size)
        name = line.split('/')[0]
        if name not in names:
            names.append(name)
        label = names.index(name)
        r = (img.shape[0] - crop_size) // 2
        c = (img.shape[1] - crop_size) // 2
        x_train.append(img[r:r+crop_size, c:c+crop_size])
        if x_train[-1].shape != (200, 200, 3): print line, img.shape
        y_train.append(label)
    for line in open(test_list_file, 'r').readlines():
        img = load(osp.join(dir, line.rstrip('\n')))
        name = line.split('/')[0]
        if name not in names:
            names.append(name)
        label = names.index(name)
        r = (img.shape[0] - crop_size) / 2
        c = (img.shape[1] - crop_size) / 2
        x_test.append(img[r:r+crop_size, c:c+crop_size])
        if x_train[-1].shape != (200, 200, 3): print line, img.shape
        y_test.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)
    np.save(x_train_file, x_train)
    np.save(y_train_file, y_train)
    np.save(x_test_file, x_test)
    np.save(y_test_file, y_test)
    return (x_train, y_train), (x_test, y_test)


def load_mnist(mode):
    if mode == 'basic':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.
    elif mode == 'rot':
        x_train, y_train = pickle.load(open('rotation_train.pkl', 'rb'))
        x_test, y_test = pickle.load(open('rotation_test.pkl', 'rb'))
    elif mode == 'rndback':
        x_train, y_train = pickle.load(open('rndback_train.pkl', 'rb'))
        x_test, y_test = pickle.load(open('rndback_test.pkl', 'rb'))
    elif mode == 'imgback':
        x_train, y_train = pickle.load(open('imgback_train.pkl', 'rb'))
        x_test, y_test = pickle.load(open('imgback_test.pkl', 'rb'))
    elif mode == 'imgback_rot':
        x_train, y_train = pickle.load(open('imgback_rot_train.pkl', 'rb'))
        x_test, y_test = pickle.load(open('imgback_rot_test.pkl', 'rb'))
    else:
        raise NotImplementedError("no this dataset")
    print x_train.max(), x_test.max()
    return (x_train, y_train), (x_test, y_test)


def clip_stich_augmentation(x, y, num):
    if num == 0: return x, y
    n = y.shape[0]
    n_classes = y.shape[1]
    images = []
    labels = []
    import cv2
    h = 32
    r = h // 2
    s = int(0.5 * r)
    for i in range(num):
        idx = np.random.randint(low=0, high=n, size=2)
        pathes = [x[i_] for i_ in idx]
        img_new = np.zeros((h, h, 3), dtype='float32')
        img_new[:, 0:r] = pathes[0][:, s:s + r]
        img_new[:, r:2*r] = pathes[1][:, s:s + r]
        # img_new[:, 2*r:3*r] = x[i, :, s:s + r]
        # img_new[:, 3*r:4*r] = x[i, :, s:s + r]
        images.append(img_new)
        l = np.zeros((n_classes,), dtype='float32')
        for j in idx:
            t = np.argwhere(y[j] == 1)[0, 0]
            l[t] += 0.5
        labels.append(l)
    return np.concatenate([x, np.asarray(images)]), np.concatenate([y, np.asarray(labels)])


def mix_augmentation(x, y, num):
    if num == 0: return x, y
    n = y.shape[0]
    n_classes = y.shape[1]
    images = []
    labels = []
    import cv2
    h = 32
    r = h // 2
    for i in range(num):
        idx = np.random.randint(low=0, high=n, size=4)
        pathes = [cv2.resize(x[i], (r, r)) for i in idx]
        img_new = np.zeros((h, h, 3), dtype='float32')
        img_new[0:r, 0:r] = pathes[0]
        img_new[0:r, r:h] = pathes[1]
        img_new[r:h, 0:r] = pathes[2]
        img_new[r:h, r:h] = pathes[3]
        images.append(img_new)
        l = np.zeros((n_classes,), dtype='float32')
        for j in idx:
            t = np.argwhere(y[j]==1)[0, 0]
            l[t] = 1
        labels.append(l)
    return np.asarray(images), np.asarray(labels)
    # return np.concatenate([x, np.asarray(images)]), np.concatenate([y, np.asarray(labels)])


def load_data(mode, batch_size):
    (x_train, y_train), (x_test, y_test) = load_mnist(mode)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    cut = int(batch_size * (x_train.shape[0] // batch_size))
    x_train = x_train[:cut]
    y_train = y_train[:cut]
    cut = int(batch_size * (x_test.shape[0] // batch_size))
    x_test = x_test[:cut]
    y_test = y_test[:cut]
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    save_cifar_to_hdf5()