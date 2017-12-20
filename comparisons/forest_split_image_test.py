
# coding: utf-8

# 
# Implementation of a random forest regressor on the split images dataset, to be used as a comparison to the
# CNN methods.

# In[1]:

# import tensorflow as tf
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


# In[3]:

class Data:
	def __init__(self, noise, size, imageFiles='./data/noise_0/train_data/reshaped/*.png', labelFiles='./data/noise_0/train_data/reshaped/*.npy'):
		image_list = glob.glob(imageFiles)
		image_list.sort()
#         print(image_list)
		flag = False
		if len(image_list) == 1:
			flag = True
			image_list=[imageFiles]
		
		x = []
		for fname in image_list:
			for img in process_image(fname):
				x.append(img)
		x = np.array(x)
		label_list = glob.glob(labelFiles)
		label_list.sort()
		if len(label_list) == 0:
			label_list=[labelFiles]
		y = np.array([])
		for l in label_list:
			y = np.concatenate((y, np.load(l)))
		y = np.reshape(y, (y.shape[0], 1))
		if not flag:
			self.x, self.y = shuffle(x, y)
		else:
			self.x, self.y = x, y
		if noise:
			self.x = np.clip(self.x + np.random.normal(0, noise*255, self.x.shape), 0, 255)
		self.x = self.x[:200*size]
		self.y = self.y[:200*size]
	
		
	def next_batch(self, batch_size):
		self.x, self.y = shuffle(self.x, self.y)
		for i in range(0, self.x.shape[0], batch_size):
			yield self.x[i:i+batch_size], self.y[i:i+batch_size]
	
	def next_batch_no_shuffle(self, batch_size):
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
			images.append(cropped[(y - 1) * img_height: y * img_height, x * img_width:(x + 1) * img_width, :].flatten())

	images = np.array(images)
	return images



def train_diff_levels(noise, size):
	data = Data(noise, size, imageFiles='./data/noise_0/train_data/reshaped/*.png', labelFiles='./data/noise_0/train_data/reshaped/*.npy')
	rf = RandomForestRegressor(n_estimators=10, verbose=5)
	rf.fit(data.x, data.y.flatten())
	pickle.dump(rf, open("saved_models/forest/split_image/noise_{0}_training_{1}.ckpt".format(noise, size), 'wb'))

def test_model(model_noise, model_training_size, image_noise):
	print("=========================================================")
	rf = pickle.load(open("saved_models/forest/split_image/noise_{0}_training_{1}.ckpt".format(model_noise, model_training_size), 'rb'))
	print("Model restored.")
	test_data = Data(image_noise, 200, imageFiles='./data/noise_0/test_data/reshaped/*.png', labelFiles='./data/noise_0/test_data/reshaped/*.npy')
	test_abs_difference = np.abs(rf.predict(test_data.x) - test_data.y.flatten())
	print("Trained with noise {0} and model_training size {1} and tested with noise {2}".format(model_noise, model_training_size, image_noise))
	print("Mean:", np.mean(test_abs_difference))
	print("25th Percentile:", np.percentile(test_abs_difference, 25))
	print("50th percentile:", np.percentile(test_abs_difference, 50))
	print("75th Percentile:", np.percentile(test_abs_difference, 75))
	print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
	print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))


def test_sim_noise_model(model_noise, sim_noise):
	print("=========================================================")
	rf = pickle.load(open("saved_models/forest/split_image/sim_noise_{0}.ckpt".format(model_noise), 'rb'))
	print("Model restored.")
	test_data = Data(0, 200, imageFiles='./data/noise_{0}/test_data/reshaped/*.png'.format(sim_noise), labelFiles='./data/noise_{0}/test_data/reshaped/*.npy'.format(sim_noise))
	test_abs_difference = np.abs(rf.predict(test_data.x) - test_data.y.flatten())
	print("Trained with sim noise {0} and tested with sim noise {1}".format(model_noise, sim_noise))
	print("25th Percentile:", np.percentile(test_abs_difference, 25))
	print("50th percentile:", np.percentile(test_abs_difference, 50))
	print("75th Percentile:", np.percentile(test_abs_difference, 75))
	print("Bottom error bar:", np.percentile(test_abs_difference, 50) - np.percentile(test_abs_difference, 25))
	print("Top error bar:", np.percentile(test_abs_difference, 75) - np.percentile(test_abs_difference, 50))


# with tf.Session() as sess:
#==============================Robustness to Image Noise===================================
# print("Trained with noise and tested with noise:")
# for noise in [0, 0.25, 0.5, 0.75]:
#     test_model(noise, 1000, noise)

# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print("Trained with noise 0.25 and tested with noise:")
# for noise in [0, 0.25, 0.5, 0.75]:
#     test_model(0.25, 1000, noise)


#============================Varying training size==========================================

# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# for size in [200, 400, 600, 800, 1000]:
#     test_model(0, size, 0)


#=============================Varying simulation noise===========================================
# print("Trained with sim_noise and tested with sim_noise")
# for sim_noise in [0.25, 0.5, 0.75, 1]:
#     test_sim_noise_model(sim_noise, sim_noise)

# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print("Trained with sim noise 0.25 and tested with noise")
# for sim_noise in [0, 0.25, 0.5, 0.75, 1]:
#     test_sim_noise_model(0.25, sim_noise)


test_model(0, 1000, 0)




