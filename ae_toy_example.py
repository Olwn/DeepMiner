import os

import tensorflow as tf
import numpy as np
import math
#import pandas as pd
#import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from dataset import load_data
from matplotlib import pyplot as plt

batch_size = 128
(x_train, y_train), (x_test, y_test) = load_data(mode='basic', batch_size=batch_size)

print x_train.shape

# Autoencoder with 1 hidden layer
n_sample, n_input = x_train.shape
n_hidden = 200

x = tf.placeholder("float", [None, n_input])
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
# h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
h = tf.nn.sigmoid(tf.matmul(x, Wh) + bh)
# Weights and biases to hidden layer
Wo = tf.transpose(Wh)
bo = tf.Variable(tf.zeros([n_input]))
# y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
y = tf.nn.sigmoid(tf.matmul(h,Wo) + bo)

# Objective functions
y_ = tf.placeholder("float", [None, n_input])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
mse = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 10000

for i in range(n_rounds):
    train_sample = np.random.randint(x_train.shape[0], size=batch_size)
    test_sample = np.random.randint(x_test.shape[0], size=batch_size)
    train_batch_x = x_train[train_sample][:] + np.random.normal(0.5, 0.1, size=(batch_size, n_input))
    train_batch_y = x_train[train_sample][:]
    test_batch_x = x_test[test_sample][:]
    test_batch_y = x_test[test_sample][:]
    sess.run(train_step, feed_dict={x: train_batch_x, y_: train_batch_y})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: test_batch_x, y_: test_batch_y})

dir_name = "sda/exp" + datetime.now().strftime("%m%d-%H-%M-%S")
if not os.path.exists(dir_name): os.mkdir(dir_name)
filters = sess.run(Wh)
plt.axis('off')
r = min(int(math.sqrt(n_hidden)), 100)
f, axarr = plt.subplots(r, r)
for i in range(r):
    for j in range(r):
        idx = i*r+j
        img = filters[:, idx].reshape((28, 28))
        axarr[i, j].imshow(img, cmap=plt.cm.get_cmap('gray'))
        axarr[i, j].axis('off')
f.savefig(os.path.join(dir_name, 'filters.jpg'))

"""
print "Final activations:"
print sess.run(y, feed_dict={x: x_train})
print "Final weights (input => hidden layer)"
print sess.run(Wh)
print "Final biases (input => hidden layer)"
print sess.run(bh)
print "Final biases (hidden layer => output)"
print sess.run(bo)
print "Final activations of hidden layer"
print sess.run(h, feed_dict={x: x_train})
"""
