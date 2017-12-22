""" Utility function for making drainage rate predictions given an aerial image. """

import tensorflow as tf
import scipy.misc
import numpy as np
import random
import glob, os

# Conv2D wrapper, with bias and REUL activation.
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# MaxPool2D wrapper.
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Entire CNN architecture.
def conv_net(x, weights, biases, dropout):
    # Convolution layer.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max pooling (down-sampling).
    conv1 = maxpool2d(conv1, k=2)

    # Convolution layer.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max pooling (down-sampling).
    conv2 = maxpool2d(conv2, k=2)

    # Convolution layer.
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    
    # Max pooling (down-sampling).
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer.
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply dropout.
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output (predictions).
    out = tf.add(tf.matmul(fc1, weights['w_out']), biases['b_out'])

    return out

# Returns the vector of drainage rate predictions after running the image from the specified path
# through the trained CNN over the entire image.
def predictions(img):
    # Input image dimensions.
    in_height = 320
    in_width = 320

    n_input = in_height * in_width  # Total number of inputs.
    n_outputs = 200 # Number of drainage rate predictions (number of plants in an image).
    dropout = 0.75 # Dropout, probability to keep units

    # Input tensor.
    x = tf.placeholder(tf.float32, [None, in_height, in_width, 3], name='x_input')

    # Output tensor.
    y = tf.placeholder(tf.float32, [None, n_outputs], name='y_output')

    # Keep probability for implementing dropout.
    keep_prob = tf.placeholder(tf.float32)

    # Weights initialized from zero-mean Gaussians with this standard deviation.
    sd = 0.01

    # Store layer weights and biases.
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=sd)),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=sd)),
        # 5x5 conv, 64 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=sd)),
        # fully connected, (320/8)*(320/8)*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([40*40*64, 1024], stddev=sd)),
        # 1024 inputs, 200 outputs
        'w_out': tf.Variable(tf.random_normal([1024, n_outputs], stddev=sd))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([64])),
        'bd1': tf.Variable(tf.zeros([1024])),
        'b_out': tf.Variable(tf.zeros([n_outputs]))
    }

    # Prediction tensor.
    pred = conv_net(x, weights, biases, keep_prob)

    # Initialize global variables.
    init = tf.global_variables_initializer()

    # Dictionary of mapping from names to tensors (makes saving and restoring models easier).
    combined = {}
    combined.update(weights)
    combined.update(biases)

    # For saving models.
    saver = tf.train.Saver(combined, max_to_keep=None)

    with tf.Session() as sess:
        sess.run(init)

        # Restore model.
        saver.restore(sess, "./saved_models/whole_image/noise_0_training_1000.ckpt")

        # Read in image.
        img = np.array(scipy.misc.imread(img, mode='RGB'))

        # Run inference.
        predictions = sess.run(pred, feed_dict={x: img.reshape((1, 320, 320, 3)), keep_prob: 1.})

        return predictions.reshape(200)
