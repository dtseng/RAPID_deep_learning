# SVR implementation.

import tensorflow as tf
import scipy.misc
import numpy as np
import random
import glob, os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import pickle
import sys

def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

class Data:
    def __init__(self, noise, size, imageFiles='./datasets/noise_0/train_data/regular/*.png', labelFiles='./datasets/noise_0/train_data/regular/*.npy'):
        image_list = glob.glob(imageFiles)
        image_list.sort()
        x = np.array([scipy.misc.imread(fname, mode='RGB').flatten() for fname in image_list])
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
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
    
    def next_batch_no_shuffle(self, batch_size):
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]


def train_diff_levels(noise, size):
    data = Data(noise, size, imageFiles='./datasets/noise_0/train_data/regular/*.png', labelFiles='./datasets/noise_0/train_data/regular/*.npy')
    svr = LinearSVR(tol=1.0, verbose=10)
    multi_svr = MultiOutputRegressor(svr)
    multi_svr.fit(data.x / 255.0, data.y[:,:2])
    pickle.dump(multi_svr, open("saved_models/svr/noise_{0}_training_{1}.ckpt".format(noise, size), 'wb'))

def test_model(model_noise, model_training_size, image_noise):
    print("=========================================================")
    multi_svr = pickle.load(open("saved_models/svr/noise_{0}_training_{1}.ckpt".format(model_noise, model_training_size), 'rb'))
    print("Model restored.")
    test_data = Data(image_noise, 200, imageFiles='./datasets/noise_0/test_data/regular/*.png', labelFiles='./datasets/noise_0/test_data/regular/*.npy')
    test_abs_difference = np.abs(multi_svr.predict(test_data.x / 255.0).flatten() - test_data.y[:,:2].flatten())
    print("Trained with noise {0} and model_training size {1} and tested with noise {2}".format(model_noise, model_training_size, image_noise))
    print("Mean:", np.mean(test_abs_difference))
    print("25th Percentile:", np.percentile(test_abs_difference, 25))
    print("50th percentile:", np.percentile(test_abs_difference, 50))
    print("75th Percentile:", np.percentile(test_abs_difference, 75))
    print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
    print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))

train_diff_levels(0, 1000)
test_model(0, 1000, 0)
