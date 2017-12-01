import os
import numpy as np
import pickle


def save_to_npy(path_from, path_to):
    images = []
    labels = []
    for line in open(path_from, 'r').readlines():
        nums = [float(x) for x in line.split()]
        images.append(nums[:-1])
        labels.append(int(nums[-1]))
    images = np.asarray(images)
    labels = np.asarray(labels)
    c = [images, labels]
    pickle_file = open(path_to, 'wb')
    pickle.dump(c, pickle_file)


def generate_rotated():
    dir_name = "/home/x/Downloads/mnist_data"
    train_file = os.path.join(dir_name, 'mnist_all_rotation_normalized_float_train_valid.amat')
    test_file = os.path.join(dir_name, 'mnist_all_rotation_normalized_float_test.amat')
    save_to_npy(train_file, 'rotation_train.pkl')
    save_to_npy(test_file, 'rotation_test.pkl')


def generate_rndback():
    dir_name = "/home/x/Downloads/mnist_data"
    train_file = os.path.join(dir_name, 'mnist_background_random_train.amat')
    test_file = os.path.join(dir_name, 'mnist_background_random_test.amat')
    save_to_npy(train_file, 'rndback_train.pkl')
    save_to_npy(test_file, 'rndback_test.pkl')


def genarate_imgback():
    dir_name = "/home/x/Downloads/mnist_data"
    train_file = os.path.join(dir_name, 'mnist_background_images_train.amat')
    test_file = os.path.join(dir_name, 'mnist_background_images_test.amat')
    save_to_npy(train_file, 'imgback_train.pkl')
    save_to_npy(test_file, 'imgback_test.pkl')


def generate_imgback_rot():
    dir_name = "/home/x/Downloads/mnist_data"
    train_file = os.path.join(dir_name, 'mnist_all_background_images_rotation_normalized_train_valid.amat')
    test_file = os.path.join(dir_name, 'mnist_all_background_images_rotation_normalized_test.amat')
    save_to_npy(train_file, 'imgback_rot_train.pkl')
    save_to_npy(test_file, 'imgback_rot_test.pkl')

if __name__ == '__main__':
    # generate_rotated()
    # genarate_imgback()
    # generate_rndback()
    generate_imgback_rot()
