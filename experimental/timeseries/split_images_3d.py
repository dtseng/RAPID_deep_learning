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
    def __init__(self, imageDir='./train_data/reshaped/', 
                 labelDir='./train_data/reshaped/', num_examples=1000):
        x = []
        for i in range(num_examples):
            if i % 100 == 0:
                logging.info("Generating example " + str(i) + " images.")
            for j in range(200):
                x.append([])
            for j in [3, 5, 7, 9]:
                image_file = imageDir + 'image' + str(i) + '-' + str(j) + '.png'
                imgs = process_image(image_file)
                for k in range(200):
                    x[i * 200 + k].append(imgs[k])
        x = np.array(x)
        
        label_list = []
        for i in range(num_examples):
            label_list.append(labelDir + 'drain_rate' + str(i) + '.npy')
        y = np.array([])
        for l in label_list:
            y = np.concatenate((y, np.load(l)))
        y = np.reshape(y, (y.shape[0], 1))
        self.x, self.y = shuffle(x, y)
        
    def next_batch(self, batch_size):
        self.x, self.y = shuffle(self.x, self.y)
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
            
        
def process_image(fname):
    img = scipy.misc.imread(fname, mode='RGB')
    cropped = img[30:430, 10:410, :]

    img_height = 20
    img_width = 40
    
    images = []

    for y in range(20, 0, -1):
        for x in range(10):
            images.append(cropped[(y - 1) * img_height: y * img_height, x * img_width:(x + 1) * img_width, :])

    images = np.array(images)
    return images

in_depth = 4
in_height = 20
in_width = 40

learning_rate = 0.001
training_iters = 1000000
batch_size = 200 * 10
display_step = 10
save_interval = 5 # number of batches per save

n_input = in_height*in_width
n_outputs = 1
dropout = 0.75 # Dropout, probability to keep units


x = tf.placeholder(tf.float32, [None, in_depth, in_height, in_width, 3], name='x_input')
y = tf.placeholder(tf.float32, [None, n_outputs], name='y_output')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def conv3d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool3d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=2)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['w_out']), biases['b_out'])
    return out

sd = 0.025

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([2, 5, 5, 3, 32], stddev=sd)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([2, 5, 5, 32, 64], stddev=sd)),
    # 5x5 conv, 64 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([1*10*5*64, 1024], stddev=sd)),
    # 1024 inputs, 10 outputs (class prediction)
    'w_out': tf.Variable(tf.random_normal([1024, n_outputs], stddev=sd))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
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

logging.basicConfig(filename='split_images_timeseries_3d.log',level=logging.INFO)

with tf.Session() as sess:
    logging.info("Initialized session")
    data = Data(imageDir='./train_data/reshaped/', 
                labelDir='./train_data/reshaped/', num_examples=1000)
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
        if step % 20000 == 0:
            logging.info("Predictions: " + str(sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.})[:10]))
            logging.info("Actual: " + str(batch_y[:10]))
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            logging.info("Minibatch Loss={:.6f}".format(loss))
    
    logging.info("Done training!")
    
    save_path = saver.save(sess, "./saved_models/split_images_timeseries_3d/split_images_timeseries_3d.ckpt")
    logging.info("Saved model.")

    test_data = Data(imageDir='./train_data/reshaped/', 
                     labelDir='./train_data/reshaped/',
                     num_examples=15)
    test_batch = test_data.next_batch(15*200)
    
    test_x, test_y = next(test_batch)
    
    predictions = sess.run(pred, feed_dict={x: test_x, keep_prob: 1.})
    
    differences = predictions - test_y
    logging.info("Average Difference:" + str(np.mean(np.absolute(differences))))
