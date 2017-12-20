import tensorflow as tf
import scipy.misc
import numpy as np
import random
import logging

def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

class Data:
    def __init__(self, imageDir='./weather_data/train_data/regular/', 
                 labelDir='./weather_data/train_data/regular/', num_examples=1000):
        image_list = []
        for i in range(num_examples):
            image_list.append(imageDir + 'image' + str(i) + '.png')
        x = np.array([scipy.misc.imread(fname, mode='RGB') for fname in image_list])
        
        label_list = []
        for i in range(num_examples):
            label_list.append(labelDir + 'drain_rate' + str(i) + '.npy')
        y = np.array([np.load(l) for l in label_list])
#         self.x, self.y = x, y
        self.x, self.y = shuffle(x, y)
        
    def next_batch(self, batch_size):
        self.x, self.y = shuffle(self.x, self.y)
        for i in range(0, self.x.shape[0], batch_size):
#             yield self.x[i], self.y[i]
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]

in_height = 320
in_width = 320

learning_rate = 0.0001
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

cost = tf.reduce_mean(tf.square(tf.subtract(pred, y)))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

combined = {}
combined.update(weights)
combined.update(biases)
saver = tf.train.Saver(combined)

logging.basicConfig(filename='weather_whole_baseline.log',level=logging.INFO)

with tf.Session() as sess:
    logging.info("Initialized session")
    data = Data(imageDir='./weather_data/train_data/regular/', 
                labelDir='./weather_data/train_data/regular/', num_examples=1000)
    newbatch = data.next_batch(batch_size)
    sess.run(init)

    logging.info("Initialized run")
    
    # Keep training until reach max iterations
    for step in range(0, training_iters, batch_size):     
        logging.info("Starting iteration number: {0}".format(step))
        try:
            batch_x, batch_y = next(newbatch)
        except StopIteration:
            newbatch = data.next_batch(batch_size)
            batch_x, batch_y = next(newbatch)
            
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step % 2000 == 0:
            logging.info("Predictions: " + str(sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.})[:10]))
            logging.info("Actual: " + str(batch_y[:10]))
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            logging.info("Minibatch Loss={:.6f}".format(loss))
            
    logging.info("Done training!")

    test_data = Data(imageDir='./weather_data/test_data/regular/', 
                     labelDir='./weather_data/test_data/regular/',
                     num_examples=200)
    test_batch = test_data.next_batch(200)
    
    test_x, test_y = next(test_batch)
    
    predictions = sess.run(pred, feed_dict={x: test_x, keep_prob: 1.})
    
    differences = predictions - test_y 
    logging.info("Average Difference:" + str(np.mean(np.absolute(differences))))

    save_path = saver.save(sess, "./saved_models/weather_whole_baseline/weather_whole_baseline.ckpt")
    logging.info("Saved model.")