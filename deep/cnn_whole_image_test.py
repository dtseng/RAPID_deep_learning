
# coding: utf-8

# In[1]:

import tensorflow as tf
import scipy.misc
import numpy as np
import random
import glob, os
import time
import sys
from scipy.stats import iqr
# np.set_printoptions(threshold=np.inf)




def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

class Data:
    def __init__(self, noise, size, imageFiles='./data/noise_0/train_data/regular/*.png', labelFiles='./data/noise_0/train_data/regular/*.npy'):
        image_list = glob.glob(imageFiles)
        image_list.sort()
        x = np.array([scipy.misc.imread(fname, mode='RGB') for fname in image_list])
        label_list = glob.glob(labelFiles)
        label_list.sort()
        y = np.array([np.load(l) for l in label_list])
        self.x, self.y = x, y
        self.x, self.y = shuffle(self.x, self.y)
        self.x = self.x[:size]
        self.y = self.y[:size]

        if noise:
            self.x = np.clip(self.x + np.random.normal(0, noise*255, self.x.shape), 0, 255)
        
    def next_batch(self, batch_size):
        self.x, self.y = shuffle(self.x, self.y)
        for i in range(0, self.x.shape[0], batch_size):
#             yield self.x[i], self.y[i]
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
    
    def next_batch_no_shuffle(self, batch_size):
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]            

in_height = 320
in_width = 320

learning_rate = 0.0001
training_iters = 50000
batch_size = 25
val_test_batch_size = 200
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
difference = tf.subtract(pred, y)
mean_difference = tf.reduce_mean(difference)
cost = tf.reduce_mean(tf.square(difference))
abs_difference = tf.abs(difference)
abs_error_mean, abs_error_variance = tf.nn.moments(tf.reshape(abs_difference, [-1]), axes = [0])
abs_error_stdev = tf.sqrt(abs_error_variance)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

combined = {}
combined.update(weights)
combined.update(biases)


normalization_mean = None
normalization_stdev = None

def normalize(x, mean, stdev):
    ret_val = x - mean
    return np.nan_to_num(ret_val/stdev)


saver = tf.train.Saver(combined, max_to_keep=None)


def train_diff_levels(noise, size):
    with tf.Session() as sess:
        print("Initialized session")
        data = Data(noise, size, imageFiles='./data/noise_0/train_data/regular/*.png', labelFiles='./data/noise_0/train_data/regular/*.npy')
        normalization_mean = np.mean(data.x, axis=0)
        normalization_stdev = np.std(data.x, axis=0)
        # data.x = normalize(data.x, normalization_mean, normalization_stdev)

        newbatch = data.next_batch(batch_size)
        sess.run(init)

 
        print("Initialized run")
        # Keep training until reach max iterations
        for step in range(0, training_iters, batch_size):     
            print("Starting iteration number: {0}".format(step))
            try:
                batch_x, batch_y = next(newbatch)
            except StopIteration:
                newbatch = data.next_batch(batch_size)
                batch_x, batch_y = next(newbatch)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            if step % 1000 == 0:
                print("Predictions: ", sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.}))
                print("Actual: ", batch_y)
                train_loss, train_abs_error_mean, train_abs_error_stdev = sess.run([cost, abs_error_mean, abs_error_stdev], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print("Minibatch Loss={:.6f}".format(train_loss))
                print("Minibatch Abs Loss={:.6f}".format(train_abs_error_mean))
                print("Minibatch Abs Stdev={:.6f}".format(train_abs_error_stdev))


                val_data = Data(noise, 200, imageFiles='./data/noise_0/val_data/regular/*.png', labelFiles='./data/noise_0/val_data/regular/*.npy')
                # val_data.x = normalize(val_data.x, normalization_mean, normalization_stdev)

                val = val_data.next_batch(val_test_batch_size)
                val_x, val_y = next(val)
                val_loss, val_abs_error_mean, val_abs_error_stdev = sess.run([cost, abs_error_mean, abs_error_stdev], feed_dict={x: val_x, y: val_y, keep_prob: 1.})
                print("Validation Loss={:.6f}".format(val_loss))
                print("Validation Abs Loss={:.6f}".format(val_abs_error_mean))
                print("Validation Abs Stdev={:.6f}".format(val_abs_error_stdev))

                save_path = saver.save(sess, "./saved_models/whole_image/noise_{0}_training_{1}.ckpt".format(noise, size))
                print("Saved model.")

        print("Done training!")

        
def test_model(sess, model_noise, model_training_size, image_noise):
    print("=========================================================")

    saver.restore(sess, "./saved_models/whole_image/noise_{0}_training_{1}.ckpt".format(model_noise, model_training_size))
    print("Model restored.")
    test_data = Data(image_noise, 200, imageFiles='./data/noise_0/test_data/regular/*.png', labelFiles='./data/noise_0/test_data/regular/*.npy')
    test_batch = test_data.next_batch_no_shuffle(val_test_batch_size)
    test_x, test_y = next(test_batch)
    test_loss, test_abs_error_mean, test_abs_error_stdev, test_abs_difference = sess.run([cost, abs_error_mean, abs_error_stdev, abs_difference], feed_dict={x: test_x, y: test_y, keep_prob: 1.})
    print("Trained with noise {0} and model_training size {1} and tested with noise {2}".format(model_noise, model_training_size, image_noise))

    print("Test Loss={:.6f}".format(test_loss))
    print("Test Abs Loss={:.6f}".format(test_abs_error_mean))
    print("Test Abs Stdev={:.6f}".format(test_abs_error_stdev))
    print("25th Percentile:", np.percentile(test_abs_difference, 25))
    print("50th percentile:", np.percentile(test_abs_difference, 50))
    print("75th Percentile:", np.percentile(test_abs_difference, 75))
    print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
    print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))

       
def test_sim_noise_model(sess, model_noise, sim_noise):
    print("=========================================================")

    saver.restore(sess, "./saved_models/whole_image/sim_noise_{0}.ckpt".format(model_noise))
    print("Model restored.")
    test_data = Data(0, 200, imageFiles='./data/noise_{0}/test_data/regular/*.png'.format(sim_noise), labelFiles='./data/noise_{0}/test_data/regular/*.npy'.format(sim_noise))
    test_batch = test_data.next_batch_no_shuffle(val_test_batch_size)
    test_x, test_y = next(test_batch)
    test_loss, test_abs_error_mean, test_abs_error_stdev, test_abs_difference = sess.run([cost, abs_error_mean, abs_error_stdev, abs_difference], feed_dict={x: test_x, y: test_y, keep_prob: 1.})
    print("Trained with noise {0} tested with noise {1}".format(model_noise, sim_noise))

    print("Test Loss={:.6f}".format(test_loss))
    print("Test Abs Loss={:.6f}".format(test_abs_error_mean))
    print("Test Abs Stdev={:.6f}".format(test_abs_error_stdev))
    print("25th Percentile:", np.percentile(test_abs_difference, 25))
    print("50th percentile:", np.percentile(test_abs_difference, 50))
    print("75th Percentile:", np.percentile(test_abs_difference, 75))
    print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
    print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))
 

with tf.Session() as sess:

    # #==============================Robustness to Image Noise===================================
    # print("Trained with noise and tested with noise:")
    # for noise in [0, 0.25, 0.5, 0.75]:
    #     test_model(sess, noise, 1000, noise)

    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print("Trained with noise 0.25 and tested with noise:")
    # for noise in [0, 0.25, 0.5, 0.75]:
    #     test_model(sess, 0.25, 1000, noise)


#============================Varying training size==========================================

    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # for size in [200, 400, 600, 800, 1000]:
    #     test_model(sess, 0, size, 0)


#=============================Varying simulation noise===========================================
    print("Trained with sim_noise and tested with sim_noise")
    for sim_noise in [0.25, 0.5, 0.75, 1]:
        test_sim_noise_model(sess, sim_noise, sim_noise)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Trained with sim noise 0.25 and tested with noise")
    for sim_noise in [0, 0.25, 0.5, 0.75, 1]:
        test_sim_noise_model(sess, 0.25, sim_noise)




# noise = 0.5
# train_diff_levels(noise, 1000)

# # for noise in [0.25, 0.5, 0.75, 1]:
# #     try:
# #         train_diff_levels(noise, 1000)
# #     except KeyboardInterrupt:
# #         raise
# #     except:
# #         continue


# for training_size in [200, 400, 600, 800, 1000]:
#     try:
#         train_diff_levels(0, training_size)
#     except KeyboardInterrupt:
#         raise
#     except:
#         continue