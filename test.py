import cv2
import numpy as np
from preprocess import flip, column_mask
import tensorflow as tf
from tensorflow.python.framework import dtypes


def column_mask_show(path):
    x = cv2.imread(path)
    x = column_mask(width=8, spaces=8)(x)
    cv2.namedWindow("win")
    cv2.imshow("win", x)
    cv2.waitKey(100000)


def flip_show(path):
    img = cv2.imread(path)
    # img = flip(patch_size=2, p=0.5, option=0)(img)
    img = flip(patch_size=2, p=0.5, option=0)(img)
    img = flip(patch_size=4, p=0.4, option=0)(img)
    img = flip(patch_size=8, p=0.3, option=0)(img)
    img = flip(patch_size=16, p=0.1, option=0)(img)
    cv2.namedWindow("win")
    cv2.imshow("win", img)
    cv2.waitKey(100000)

# flip_show("/Users/O/Downloads/1.jpg")
# column_mask_show("/Users/O/Downloads/1.jpg")
h = 10
m = 1
a = tf.ones([10, 10], dtype=dtypes.int32)
c = tf.Variable(tf.ones([10, 10], dtype=dtypes.int32))
for i in range(50):
    t1 = tf.random_uniform([1], minval=0, maxval=10 - m, dtype=dtypes.int32)[0]
    t2 = tf.random_uniform([1], minval=0, maxval=10 - m, dtype=dtypes.int32)[0]
    b = tf.ones([m, m], dtype=dtypes.int32)
    b = tf.pad(b, [[t1, 10-m-t1], [t2, 10 - m-t2]])
    a = tf.multiply(1-b, a)

sess = tf.Session()
for i in range(1):
    # print sess.run([tf.multiply(a, 1-b)])
    v1 = sess.run([tf.multinomial(tf.log([[0.3, 0.7]]), 100)])
    print v1[0].reshape(10, 10)
    print np.random.choice()
