import tensorflow as tf
import scipy.misc
import numpy as np
import random
import glob, os

def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

class Data:
    def __init__(self, imageFiles='./data/noise_0/train_data/regular/*.png', labelFiles='./data/noise_0/train_data/regular/*.npy'):
        image_list = glob.glob(imageFiles)
        image_list.sort()
        x = np.array([scipy.misc.imread(fname, mode='RGB') for fname in image_list])
        label_list = glob.glob(labelFiles)
        label_list.sort()
        y = np.array([np.load(l) for l in label_list])
        self.x, self.y = x, y
        
    def next_batch(self, batch_size):
        self.x, self.y = shuffle(self.x, self.y)
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]

in_height = 320
in_width = 320

learning_rate = 0.001
training_iters = 10000
batch_size = 25
display_step = 10
save_interval = 5 # number of batches per save

n_input = in_height*in_width
n_outputs = 200 
dropout = 0.75 # Dropout, probability to keep units


x = tf.placeholder(tf.float32, [None, in_height, in_width, 3], name='x_input')
y = tf.placeholder(tf.float32, [None, n_outputs], name='y_output')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    
    # Max Pooling
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['w_out']), biases['b_out'])
    return out

def predictions(image_file_name):
    sd = 0.01
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=sd)),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=sd)),
        # 5x5 conv, 64 inputs, 256 outputs
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=sd)),
        # fully connected, (320/8)*(320/4)*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([40*40*64, 1024], stddev=sd)),
        # 1024 inputs, 10 outputs (class prediction)
        'w_out': tf.Variable(tf.random_normal([1024, n_outputs], stddev=sd))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([64])),
        'bd1': tf.Variable(tf.zeros([1024])),
        'b_out': tf.Variable(tf.zeros([n_outputs]))
    }

    pred = conv_net(x, weights, biases, keep_prob)

    init = tf.global_variables_initializer()

    combined = {}
    combined.update(weights)
    combined.update(biases)
    saver = tf.train.Saver(combined)

    with tf.Session() as sess:
    
        sess.run(init)

        img = np.array(scipy.misc.imread(image_file_name, mode='RGB'))

        saver.restore(sess, "./saved_models/whole_image_noise_0/whole_image_noise_0.ckpt")  
        predictions = sess.run(pred, feed_dict={x: img.reshape((1, 320, 320, 3)), keep_prob: 1.})

    return predictions.reshape(200)