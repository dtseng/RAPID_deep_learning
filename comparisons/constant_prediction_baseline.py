""" Simple baseline for comparison that predicts the mean drainage rate over
all training examples. """

import scipy.misc
import numpy as np
import random
import glob, os
import time

# Load drainage rate files and compute the mean.
labelFiles = './data/noise_0/train_data/regular/*.npy'
label_list = glob.glob(labelFiles)
label_list.sort()
y = np.array([np.load(l) for l in label_list]).flatten()
mean_drainage = np.mean(y)

# Load test labels.
labelFiles = './data/noise_0/test_data/regular/*.npy'
label_list = glob.glob(labelFiles)
label_list.sort()
y = np.array([np.load(l) for l in label_list]).flatten()

# Compute absolute value difference between predictions and ground truth.
test_abs_difference = np.abs(y - mean_drainage*np.ones(y.shape))
print("25th Percentile:", np.percentile(test_abs_difference, 25))
print("50th percentile:", np.percentile(test_abs_difference, 50))
print("75th Percentile:", np.percentile(test_abs_difference, 75))
print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))
