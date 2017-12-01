"""Tutorial on how to create a denoising autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import os
import matplotlib

from dataset import load_data

matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import math
from datetime import datetime
from keras import backend as K
from keras.losses import binary_crossentropy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
batch_size = 128
n_epochs = 100


def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2,
                                                    dtype=tf.int32), tf.float32))

def shift(x):
    shape = (x.get_shape()[0].value, 28, 28, 1)
    images = tf.reshape(x, shape)
    p = int(0.15 * 28)
    padding = tf.pad(images, [[0, 0], [p, p], [p, p], [0, 0]])
    crop = tf.random_crop(padding, size=shape)
    return tf.reshape(crop, (batch_size, 784))


def mask(x):
    shape = K.int_shape(x)
    binary_mask = tf.multinomial(tf.log([[5., 5.]]), num_samples=np.prod(shape))
    return x * tf.reshape(tf.cast(binary_mask, dtype=tf.float32), shape)


def autoencoder(dimensions=[784, 512]):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    x = tf.placeholder(tf.float32, [batch_size, dimensions[0]], name='x')

    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1. - corrupt_prob)
    shift_prob = tf.placeholder(tf.float32, [1])
    current_input = shift(current_input) * shift_prob + current_input * (1. - shift_prob)
    mask_prob = tf.placeholder(tf.float32, [1])
    current_input = mask(current_input) * mask_prob + current_input * (1. - mask_prob)

    # Build the encoder
    encoder = []
    weights = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        weights.append(W)
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=y))
    # cost = tf.reduce_mean(binary_crossentropy(y, x))
    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'shift_prob': shift_prob,
            'mask_prob': mask_prob,
            'weights': weights,
            'cost': cost}

# %%


def test_mnist():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # load MNIST as before
    (x_train, y_train), (x_test, y_test) = load_data(mode='basic', batch_size=batch_size)
    ae = autoencoder(dimensions=[784, 100])

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(0.001).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    lr = 0.01
    for epoch_i in range(n_epochs):
        for batch_i in range(x_train.shape[0] // batch_size):
            train_sample = np.random.randint(x_train.shape[0], size=batch_size)
            test_sample = np.random.randint(x_test.shape[0], size=batch_size)
            train_batch_x = x_train[train_sample][:]
            test_batch_x = x_test[test_sample][:]
            lr *= 0.95
            sess.run(optimizer, feed_dict={
                learning_rate: lr,
                ae['x']: train_batch_x,
                ae['corrupt_prob']: [1.0],
                ae['shift_prob']: [1.0],
                ae['mask_prob']: [0.0]
            })
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: test_batch_x,
            ae['corrupt_prob']: [0.],
            ae['shift_prob']: [0.],
            ae['mask_prob']: [0.0]
        }))

    dir_name = "sda/exp" + datetime.now().strftime("%m%d-%H-%M-%S")
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    filters = sess.run(ae['weights'][0])
    plt.axis('off')
    r = min(int(math.sqrt(filters.shape[1])), 100)
    f, axarr = plt.subplots(r, r, figsize=(20, 20))
    for i in range(r):
        for j in range(r):
            idx = i * r + j
            img = filters[:, idx].reshape((28, 28))
            axarr[i, j].imshow(img, cmap=plt.cm.get_cmap('gray'))
            axarr[i, j].axis('off')
    f.savefig(os.path.join(dir_name, 'filters.jpg'))

    # Plot example reconstructions
    """
    recon = sess.run(ae['y'], feed_dict={
        ae['x']: x_test[:batch_size], ae['corrupt_prob']: [0.0], ae['shift_prob']: [0.]})
    fig, axs = plt.subplots(2, batch_size, figsize=(10, 2))
    for example_i in range(batch_size):
        axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (28, 28)),
                                 cmap=plt.cm.get_cmap('gray'))
        axs[1][example_i].imshow(np.reshape([recon[example_i, :] + mean_img], (28, 28)),
                                 cmap=plt.cm.get_cmap('gray'))
    fig.savefig(os.path.join(dir_name, 'digits.jpg'))
    """

if __name__ == '__main__':
    test_mnist()