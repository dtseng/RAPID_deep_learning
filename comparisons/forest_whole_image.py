""" Implementation of a random forest regressor on whole images. """

import scipy.misc
import numpy as np
import random
import glob, os
from sklearn.ensemble import RandomForestRegressor
import pickle

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

# Training function for a single random forest model.
def train_diff_levels(img_noise, size, depth, sim_noise):
    # Load data with specified amount of noise and number of examples.
    data = Data(img_noise, size, imageFiles="./datasets/noise_{0}_alt/train_data/regular/*.png".format(sim_noise), 
                labelFiles="./datasets/noise_{0}_alt/train_data/regular/*.npy".format(sim_noise))

    # Train random forest model.
    rf = RandomForestRegressor(n_estimators=10, verbose=5, n_jobs=-1, max_depth=depth)
    rf.fit(data.x / 255.0, data.y)

    # Save trained model.
    pickle.dump(rf, open("saved_models/forest/whole_image2/imgnoise_{0}_training_{1}_depth_{2}_simnoise_{3}.ckpt".format(img_noise, size, depth, sim_noise), 'wb'))

# Load model trained with specified amount of noise and number of training examples and test on
# data with specified level of image noise or simulation noise.
def test_model(model_img_noise, model_training_size, depth, model_sim_noise, image_noise, sim_noise):
    print("=========================================================")

    # Load trained model.
    rf = pickle.load(open("saved_models/forest/whole_image2/imgnoise_{0}_training_{1}_depth_{2}_simnoise_{3}.ckpt".format(model_img_noise, model_training_size, depth, model_sim_noise), 'rb'))
    print("Model restored.")

    # Load test data.
    test_data = Data(image_noise, 200, imageFiles="./datasets/noise_{0}/test_data/regular/*.png".format(sim_noise), 
                     labelFiles="./datasets/noise_{0}/test_data/regular/*.npy".format(sim_noise))

    # Compute absolute value difference between predictions and ground truth.
    test_abs_difference = np.abs(rf.predict(test_data.x / 255.0).flatten() - test_data.y.flatten())
    print("Trained with image noise {0}, model_training size {1}, and sim noise {2} and tested with image noise noise {3} and sim noise {4}".format(model_img_noise, model_training_size, model_sim_noise, image_noise, sim_noise))
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
    MAX_DEPTH = 20

    # Main experiment.
    start_time = time.time()
    train_diff_levels(0, 1000, MAX_DEPTH, 0)
    end_time = time.time()

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Training Time in Seconds: " + str(end_time - start_time))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Training size experiments.
    for training_size in [25, 50, 100, 200, 400, 600, 800]:
        train_diff_levels(0, training_size, MAX_DEPTH, 0)

# Function for evaluating results for all models.
def evaluate_all_models():
    MAX_DEPTH = 20

    #============================Varying training size==========================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for size in [25, 100, 400, 1000, 2000]:
        median, bottom, top = test_model(0, size, MAX_DEPTH, 0, 0, 0)
        y_medians.append(median)
        y_bottom_error.append(bottom)
        y_top_error.append(top)

    print("MEDIAN:", y_medians)
    print("BOTTOM ERROR:", y_bottom_error)
    print("TOP ERROR:", y_top_error)

    #=============================Varying simulation noise===========================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Trained with sim noise 0 and tested with noise")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for sim_noise in [0, 25, 50, 75, 100]:
        median, bottom, top = test_model(0, 1000, MAX_DEPTH, 0, 0, sim_noise)
        y_medians.append(median)
        y_bottom_error.append(bottom)
        y_top_error.append(top)

    print("MEDIANS:", y_medians)
    print("BOTTOM ERRORS:", y_bottom_error)
    print("TOP ERRORS:", y_top_error)

    #==============================Robustness to Image Noise===================================
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Trained with noise 0 and tested with noise:")
    y_medians = []
    y_bottom_error = []
    y_top_error = []
    for noise in [0, 0.25, 0.5, 0.75]:
        median, bottom, top = test_model(0.25, 1000, MAX_DEPTH, 0, noise, 0)
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
