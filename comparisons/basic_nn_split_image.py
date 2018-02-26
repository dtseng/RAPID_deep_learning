""" Implementation of a convolutional neural network for prediction drainage rates
from synthetic aerial agricultural images of individual plants. """

import tensorflow as tf
import scipy.misc
import numpy as np
import random
import glob, os
import sys
import time

# Used for specifying which GPU to train on.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Utility function for shuffling training eamples and labels.
def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

# Class for reading in training examples.
class Data:
    def __init__(self, noise, size, imageFiles='./datasets/noise_0/train_data/reshaped/*.png', 
                 labelFiles='./datasets/noise_0/train_data/reshaped/*.npy'):
        # Read in images.
        image_list = glob.glob(imageFiles)
        image_list.sort()
        x = []
        for fname in image_list:
            for img in process_image(fname):
                x.append(img)
        x = np.array(x)

        # Read in labels.
        label_list = glob.glob(labelFiles)
        label_list.sort()
        y = np.array([])
        for l in label_list:
            y = np.concatenate((y, np.load(l)))
        y = np.reshape(y, (y.shape[0], 1))

        # Shuffle training examples.
        self.x, self.y = x, y
        self.x, self.y = shuffle(self.x, self.y)

        # Inject image noise if specified.
        if noise:
            self.x = np.clip(self.x + np.random.normal(0, noise*255, self.x.shape), 0, 255)

        # Truncate based on number of specified examples.
        self.x = self.x[:200*size]
        self.y = self.y[:200*size]
    
    # Returns a batch of training examples.
    def next_batch(self, batch_size):
        self.x, self.y = shuffle(self.x, self.y)
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
    
    # Returns a batch of training examples without shuffling after finishing an epoch.
    def next_batch_no_shuffle(self, batch_size):
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
            
# Process an individual image and split it into images of inividual plants. 
def process_image(fname):
    img = scipy.misc.imread(fname, mode='RGB')
    cropped = img[30:430, 10:410, :]

    img_height = 20
    img_width = 40
    
    images = []

    # Iterate through the image according to the order of the labels.
    for y in range(20, 0, -1):
        for x in range(10):
            images.append(cropped[(y - 1) * img_height: y * img_height, x * img_width:(x + 1) * img_width, :])

    images = np.array(images)
    return images

# Input image dimensions.
in_height = 20
in_width = 40

# Various training hyperparameters.
learning_rate = 0.001
batch_size = 25 * 200
val_batch_size = 200 * 200  # Number of examples in validation set.

n_input = in_height * in_width  # Total number of inputs.
n_outputs = 1   # Number of drainage rate predictions (just one for image of individual plant).
dropout = 0.75  # Dropout, probability to keep units.

# Input tensor.
x = tf.placeholder(tf.float32, [None, in_height, in_width, 3], name='x_input')

# Output tensor.
y = tf.placeholder(tf.float32, [None, n_outputs], name='y_output')

# Keep probability for implementing dropout.
keep_prob = tf.placeholder(tf.float32)

# Weights initialized from zero-mean Gaussians with this standard deviation.
sd = 0.001

# Number of units in the hidden layer.
num_hidden_units = 4096

# Store layer weights and biases.
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'w1': tf.Variable(tf.random_normal([in_height * in_width * 3, num_hidden_units], stddev=sd)),
    # 1024 inputs, 10 outputs (class prediction)
    'w_out': tf.Variable(tf.random_normal([num_hidden_units, n_outputs], stddev=sd))
}

biases = {
    'b1': tf.Variable(tf.zeros([num_hidden_units])),
    'b_out': tf.Variable(tf.zeros([n_outputs]))
}

# Hidden layer (with ReLU activation and dropout).
hidden_layer = tf.reshape(x, [-1, in_height * in_width * 3])
hidden_layer = tf.add(tf.matmul(hidden_layer, weights['w1']), biases['b1'])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, dropout)

# Prediction tensor.
pred = tf.add(tf.matmul(hidden_layer, weights['w_out']), biases['b_out'])

# Cost tensor is the squared error.
difference = tf.subtract(pred, y)
cost = tf.reduce_mean(tf.square(difference))

# Absolute value difference for reporting results.
abs_difference = tf.abs(difference)
abs_error_mean, abs_error_variance = tf.nn.moments(tf.reshape(abs_difference, [-1]), axes = [0])
abs_error_stdev = tf.sqrt(abs_error_variance)

# Optimization uses Adam.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize global variables.
init = tf.global_variables_initializer()

# Dictionary of mapping from names to tensors (makes saving and restoring models easier).
combined = {}
combined.update(weights)
combined.update(biases)

# For saving models.
saver = tf.train.Saver(combined, max_to_keep=None)

# Training function using specified amount of image noise and number of training examples.
def train_diff_levels(noise, size):
    with tf.Session() as sess:
        print("Initialized session with noise level {0}".format(noise))

        # Load training data.
        if size > 1000:
            data = Data(noise, size, imageFiles='./datasets/noise_0_alt/train_data/reshaped/*.png',
                        labelFiles='./datasets/noise_0_alt/train_data/reshaped/*.npy')
        else:
            data = Data(noise, size, imageFiles='./datasets/noise_0/train_data/reshaped/*.png',
                        labelFiles='./datasets/noise_0/train_data/reshaped/*.npy')

        # Get an iterator that returns batches of training data. 
        newbatch = data.next_batch(batch_size)

        # Initialize TensorFlow session.
        sess.run(init)
        print("Initialized run")

        training_iters = max(200 * size * 40, 200 * 1000 * 40)

        # Keep training until reach we reach maximum number of iterations.
        for step in range(0, training_iters, batch_size):  
            if step % 100000 == 0:   
                print("Starting iteration number: {0}".format(step))

            # Read in a batch.
            try:
                batch_x, batch_y = next(newbatch)
            except StopIteration:
                newbatch = data.next_batch(batch_size)
                batch_x, batch_y = next(newbatch)

            # Perform actual training.
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

            # Report results every epoch.
            if step % 400000 == 0:
                # Samples predictions vs. ground truth.
                print("Predictions: ", sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.}))
                print("Actual: ", batch_y)

                # Training statistics.
                train_loss, train_abs_error_mean, train_abs_error_stdev = sess.run([cost, abs_error_mean, abs_error_stdev], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print("Minibatch Loss={:.6f}".format(train_loss))
                print("Minibatch Abs Stdev={:.6f}".format(train_abs_error_stdev))
                print("Minibatch Abs Loss={:.6f}".format(train_abs_error_mean))

                # Read in validation data and report statistics.
                val_data = Data(noise, 200, imageFiles='./datasets/noise_0/val_data/reshaped/*.png', labelFiles='./datasets/noise_0/val_data/reshaped/*.npy')             
                val = val_data.next_batch(val_batch_size)
                val_x, val_y = next(val)
                val_loss, val_abs_error_mean, val_abs_error_stdev = sess.run([cost, abs_error_mean, abs_error_stdev], feed_dict={x: val_x, y: val_y, keep_prob: 1.})
                print("Validation Loss={:.6f}".format(val_loss))
                print("Validation Abs Stdev={:.6f}".format(val_abs_error_stdev))
                print("Validation Abs Loss={:.6f}".format(val_abs_error_mean))
        
        # Save the model.
        save_path = saver.save(sess, "./saved_models/basic_nn_split_image/noise_{0}_training_{1}.ckpt".format(noise, size))
        print("Saved model.")
        print("Done training!")

# Function for evaluating trained model using specified number of training examples. Evaluates the model
# on specified levels of image and simulation noise.
def test_model(sess, model_training_size, image_noise, sim_noise):
    print("=========================================================")

    # Restore model.
    saver.restore(sess, "./saved_models/basic_nn_split_image/noise_0_training_{0}.ckpt".format(model_training_size))
    print("Model restored.")

    # Load test data.
    test_data = Data(image_noise, 200, imageFiles="./datasets/noise_{0}/test_data/reshaped/*.png".format(sim_noise),
                     labelFiles="./datasets/noise_{0}/test_data/reshaped/*.npy".format(sim_noise))
    test_batch = test_data.next_batch_no_shuffle(val_batch_size)
    test_x, test_y = next(test_batch)
    test_loss, test_abs_error_mean, test_abs_error_stdev, test_abs_difference = sess.run([cost, abs_error_mean, abs_error_stdev, abs_difference], feed_dict={x: test_x, y: test_y, keep_prob: 1.})
    print("Trained with model_training size {0} and tested with image noise {1} and sim noise {2}".format(model_training_size, image_noise, sim_noise))

    # Compute statistics on test set.
    print("Test Loss={:.6f}".format(test_loss))
    print("Test Abs Loss={:.6f}".format(test_abs_error_mean))
    print("Test Abs Stdev={:.6f}".format(test_abs_error_stdev))
    print("25th Percentile:", np.percentile(test_abs_difference, 25))
    print("50th percentile:", np.percentile(test_abs_difference, 50))
    print("75th Percentile:", np.percentile(test_abs_difference, 75))
    print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
    print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))

    # Return error bars for plotting.
    return np.percentile(test_abs_difference, 50), np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25), np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50)
 
# Trains all models needed for experiments.
def train_all_models():
    # # Main experiment.
    start_time = time.time()
    train_diff_levels(0, 1000)
    end_time = time.time()

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Training Time in Seconds: " + str(end_time - start_time))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Training size experiment.
    for training_size in [25, 100, 400, 2000]:
        print("Training for training size ", training_size)
        train_diff_levels(0, training_size)

# Evalutes all models for experiments.
def evaluate_all_models():
    with tf.Session() as sess:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Testing on varying training set sizes:")
        y_medians = []
        y_bottom_error = []
        y_top_error = []
        for size in [25, 100, 400, 1000, 2000]:
            median, bottom, top = test_model(sess, size, 0, 0)
            y_medians.append(median)
            y_bottom_error.append(bottom)
            y_top_error.append(top)

        print("MEDIANS:", y_medians)
        print("BOTTOM ERRORS:", y_bottom_error)
        print("TOP ERRORS:", y_top_error)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Testing on simulation noise sets")
        y_medians = []
        y_bottom_error = []
        y_top_error = []
        for sim_noise in [0, 25, 50, 75, 100]:
            median, bottom, top = test_model(sess, 1000, 0, sim_noise)
            y_medians.append(median)
            y_bottom_error.append(bottom)
            y_top_error.append(top)

        print("MEDIANS:", y_medians)
        print("BOTTOM ERRORS:", y_bottom_error)
        print("TOP ERRORS:", y_top_error)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Testing on image noise sets:")
        y_medians = []
        y_bottom_error = []
        y_top_error = []
        for noise in [0, 0.25, 0.5, 0.75]:
            median, bottom, top = test_model(sess, 1000, noise, 0)
            y_medians.append(median)
            y_bottom_error.append(bottom)
            y_top_error.append(top)

        print("MEDIANS:", y_medians)
        print("BOTTOM ERRORS:", y_bottom_error)
        print("TOP ERRORS:", y_top_error)

def main():
    # train_all_models()
    evaluate_all_models()

if __name__ == '__main__':
    main()