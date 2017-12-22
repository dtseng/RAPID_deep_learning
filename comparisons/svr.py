""" Support Vector Regressor implementation for comparison. """

import scipy.misc
import numpy as np
import random
import glob, os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import pickle
import time
import sys

# Utility function for shuffling training eamples and labels.
def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

# Class for reading in training examples.
class Data:
    def __init__(self, noise, size, imageFiles='./datasets/noise_0/train_data/regular/*.png', 
                 labelFiles='./datasets/noise_0/train_data/regular/*.npy'):
        # Get images.
        image_list = glob.glob(imageFiles)
        image_list.sort()
        x = np.array([scipy.misc.imread(fname, mode='RGB').flatten() for fname in image_list])

        # Get labels.
        label_list = glob.glob(labelFiles)
        label_list.sort()
        y = np.array([np.load(l) for l in label_list])

        # Shuffle training examples.
        self.x, self.y = x, y
        self.x, self.y = shuffle(self.x, self.y)

        # Truncate based on number of specified examples.
        self.x = self.x[:size]
        self.y = self.y[:size]

        # Inject image noise if specified.
        if noise:
            self.x = np.clip(self.x + np.random.normal(0, noise*255, self.x.shape), 0, 255)

# Training function for a single SVR model.
def train_diff_levels(noise, size):
    # Load data with specified amount of noise and number of examples.
    data = Data(noise, size, imageFiles='./datasets/noise_0_alt/train_data/regular/*.png',
                labelFiles='./datasets/noise_0_alt/train_data/regular/*.npy')

    # Train the SVR.
    svr = LinearSVR(tol=0.1, verbose=10)
    multi_svr = MultiOutputRegressor(svr, n_jobs=-1)
    multi_svr.fit(data.x / 255.0, data.y)

    # Save trained model.
    pickle.dump(multi_svr, open("saved_models/svr/noise_{0}_training_{1}.ckpt".format(noise, size), 'wb'))

# Load model trained with specified amount of noise and number of training examples and test on
# data with specified level of image noise or simulation noise.
def test_model(model_noise, model_training_size, image_noise, sim_noise):
    print("=========================================================")

    # Load trained model.
    multi_svr = pickle.load(open("saved_models/svr/noise_{0}_training_{1}.ckpt".format(model_noise, model_training_size), 'rb'))
    print("Model restored.")

    # Load test data.
    test_data = Data(image_noise, 200, imageFiles="./datasets/noise_{0}/test_data/regular/*.png".format(sim_noise), 
                     labelFiles="./datasets/noise_{0}/test_data/regular/*.npy".format(sim_noise))

    # Compute absolute value difference between predictions and ground truth.
    test_abs_difference = np.abs(multi_svr.predict(test_data.x / 255.0).flatten() - test_data.y.flatten())
    print("Trained with noise {0} and model_training size {1} and tested with image noise {2} and sim noise {3}".format(model_noise, model_training_size, image_noise, sim_noise))
    print("Mean:", np.mean(test_abs_difference))
    print("25th Percentile:", np.percentile(test_abs_difference, 25))
    print("50th percentile:", np.percentile(test_abs_difference, 50))
    print("75th Percentile:", np.percentile(test_abs_difference, 75))
    print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
    print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))

    # Return error bar values for plotting.
    return np.percentile(test_abs_difference, 50), np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25), np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50)

# Training function for all experiments.
def train_all_models():
    # Training size experiment.
    for training_size in [100, 200, 400, 600, 800]:
        start_time = time.time()
        train_diff_levels(0, training_size)
        end_time = time.time()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Number of Examples: " + str(training_size))
        print("Training Time in Seconds: " + str(end_time - start_time))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# Function for evaluating results for all models.
def evaluate_all_models():
    #============================Varying training size==========================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for size in [25, 100, 400, 1000, 2000]:
        median, bottom, top = test_model(0, size, 0, 0)
        y_medians.append(median)
        y_bottom_error.append(bottom)
        y_top_error.append(top)

    print("MEDIANS:", y_medians)
    print("BOTTOM ERRORS:", y_bottom_error)
    print("TOP ERRORS:", y_top_error)

    #=============================Varying simulation noise===========================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Trained with no simulation noise and tested with noise")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for sim_noise in [0, 25, 50, 75, 100]:
        median, bottom, top = test_model(0, 1000, 0, sim_noise)
        y_medians.append(median)
        y_bottom_error.append(bottom)
        y_top_error.append(top)

    print("MEDIANS:", y_medians)
    print("BOTTOM ERRORS:", y_bottom_error)
    print("TOP ERRORS:", y_top_error)

    #==============================Robustness to Image Noise===================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Trained with noise 0.25 and tested with noise:")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for noise in [0, 0.25, 0.5, 0.75]:
        test_model(0, 1000, noise, 0)
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
