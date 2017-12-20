
# coding: utf-8

# 
# Implementation of a random forest regressor on the whole images dataset, to be used as a comparison to the
# CNN methods.

# In[1]:

import tensorflow as tf
import scipy.misc
import numpy as np
import random
# import matplotlib.pyplot as plt
import glob, os
from sklearn.ensemble import RandomForestRegressor
import pickle


# In[2]:

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
#             yield self.x[i], self.y[i]
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]
    
    def next_batch_no_shuffle(self, batch_size):
        for i in range(0, self.x.shape[0], batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]


# In[3]:



def train_diff_levels(noise, size):
    data = Data(noise, size, imageFiles='./data/noise_0/train_data/regular/*.png', labelFiles='./data/noise_0/train_data/regular/*.npy')
    rf = RandomForestRegressor(n_estimators=10, verbose=5)
    rf.fit(data.x, data.y)
    pickle.dump(rf, open("saved_models/forest/whole_image/noise_{0}_training_{1}.ckpt".format(noise, size), 'wb'))


for training_size in [200, 400, 600, 800, 1000]:
    try:
        train_diff_levels(0, training_size)
    except KeyboardInterrupt:
        raise
    except:
        continue

for noise in [0.25, 0.5, 0.75, 1]:
    try:
        train_diff_levels(noise, 1000)
    except KeyboardInterrupt:
        raise
    except:
        continue