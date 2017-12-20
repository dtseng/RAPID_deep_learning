import tensorflow as tf
import scipy.misc
import numpy as np
import random
import logging

def shuffle(X, Y, Z):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Z = Z[indices]
    return X, Y, Z

class Data:
    def __init__(self, imageDir='./weather_data/train_data/reshaped/', 
                 labelDir='./weather_data/train_data/reshaped/', num_examples=1000):
        x_image = []
        for i in range(num_examples):
            image_file = imageDir + "image" + str(i) + ".png"
            for img in process_image(image_file):
                x_image.append(img)       
        x_image = np.array(x_image)
        
        x_weather = np.array([])
        for i in range(num_examples):
            weather_file = imageDir + "weather" + str(i) + ".npy"
            weather_params = np.load(weather_file)
            x_weather = np.concatenate([x_weather, weather_params * np.ones(200)])
        x_weather = np.reshape(x_weather, (x_weather.shape[0], 1))    
              
        y = np.array([])
        for i in range(num_examples):
            label_file = labelDir + "drain_rate" + str(i) + ".npy"
            y = np.concatenate((y, np.load(label_file)))
        y = np.reshape(y, (y.shape[0], 1))
        
        self.x_image, self.x_weather, self.y = shuffle(x_image, x_weather, y)
        
    def next_batch(self, batch_size):
        self.x_image, self.x_weather, self.y = shuffle(self.x_image, self.x_weather, self.y)
        for i in range(0, self.x_image.shape[0], batch_size):
            yield self.x_image[i:i+batch_size], self.x_weather[i:i+batch_size], self.y[i:i+batch_size]

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

in_height = 20
in_width = 40

learning_rate = 0.001
training_iters = 5000000
batch_size = 10 * 200
display_step = 10
save_interval = 5 # number of batches per save

n_input = in_height*in_width
n_outputs = 1
dropout = 0.75 # Dropout, probability to keep units


x_img = tf.placeholder(tf.float32, [None, in_height, in_width, 3], name='x_img')
x_weather = tf.placeholder(tf.float32, [None, 1], name='x_weather')
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

def conv_net(x_img, x_weather, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    full_fc = tf.concat([fc1, x_weather], 1)

    # Output, class prediction
    fc_2 = tf.add(tf.matmul(full_fc, weights['wd2']), biases['bd2'])

    out = tf.add(tf.matmul(fc_2, weights['w_out']), biases['b_out'])
    return out

sd = 0.01

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=sd)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=sd)),
    # fully connected, (320/8)*(320/4)*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([10*5*64, 1024], stddev=sd)),
    # 1024 inputs, 10 outputs (class prediction)
    'wd2': tf.Variable(tf.random_normal([1025, 1024], stddev=sd)),
    'w_out': tf.Variable(tf.random_normal([1024, n_outputs], stddev=sd))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bd1': tf.Variable(tf.zeros([1024])),
    'bd2': tf.Variable(tf.zeros([1024])),
    'b_out': tf.Variable(tf.zeros([n_outputs]))
}

pred = conv_net(x_img, x_weather, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.square(tf.subtract(pred, y)))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

combined = {}
combined.update(weights)
combined.update(biases)
saver = tf.train.Saver(combined)

logging.basicConfig(filename='weather.log',level=logging.INFO)

with tf.Session() as sess:
    logging.info("Initialized session")
    data = Data(imageDir='./weather_data/train_data/reshaped/', 
                labelDir='./weather_data/train_data/reshaped/', num_examples=1000)
    newbatch = data.next_batch(batch_size)
    sess.run(init)

    logging.info("Initialized run")
    
    # Keep training until reach max iterations
    for step in range(0, training_iters, batch_size):     
        logging.info("Starting iteration number: {0}".format(step))
        try:
            batch_x_img, batch_x_weather, batch_y = next(newbatch)
        except StopIteration:
            newbatch = data.next_batch(batch_size)
            batch_x_img, batch_x_weather, batch_y = next(newbatch)
        sess.run(optimizer, feed_dict={x_img: batch_x_img, x_weather: batch_x_weather,
                                       y: batch_y, keep_prob: dropout})

        if step % 200000 == 0:
            logging.info("Predictions: " + str(sess.run(pred, feed_dict={x_img: batch_x_img, x_weather: batch_x_weather,
                                                             keep_prob: 1.})[:10]))
            logging.info("Actual: " + str(batch_y[:10]))
            loss = sess.run(cost, feed_dict={x_img: batch_x_img, x_weather: batch_x_weather,
                                             y: batch_y, keep_prob: 1.})
            logging.info("Minibatch Loss={:.6f}".format(loss))
            
    logging.info("Done training!")

    test_data = Data(imageDir='./weather_data/test_data/reshaped/', 
                     labelDir='./weather_data/test_data/reshaped/',
                     num_examples=200)
    test_batch = test_data.next_batch(200*200)
    
    test_x_img, test_x_weather, test_y = next(test_batch)
    
    predictions = sess.run(pred, feed_dict={x_img: test_x_img, x_weather: test_x_weather, keep_prob: 1.})
    
    differences = predictions - test_y 
    logging.info("Average Difference:" + str(np.mean(np.absolute(differences))))

    save_path = saver.save(sess, "./saved_models/weather/weather.ckpt")
    print("Saved model.")
