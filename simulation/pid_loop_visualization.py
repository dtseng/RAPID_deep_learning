""" Script for saving images of the vineyard over time after applying the control loop. """

import simulation
import predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf

from PIL import Image
from matplotlib.animation import FuncAnimation

# Used for specifying which GPU to train on.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# control loop noise
STD_DEV = 0
# drainage rate
RATE = 5

# Initialize example vineyard.
vy = simulation.Vineyard()
vy.drainage_rate = np.load("/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(RATE))

# Update vineyard for 10 timesteps.
for i in range(10):
    vy.update(0)

# File pattern for save images.
DIRECTORY = "pid_test_drain_rate{0}_std_{1}/".format(RATE, STD_DEV)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
IMG_FILENAME = DIRECTORY + "test_img{0}.png"

# Save current image of vineyard.
extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
vy.fig.savefig(IMG_FILENAME.format(10), bbox_inches=extent)
size = 320, 320
im = Image.open(IMG_FILENAME.format(10))
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save(IMG_FILENAME.format(10), "PNG")

# Intialize predictor.
predictor = predictions.Predictor("/home/wsong/saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())

total_irrigation_used = 0.0

# set value
SET_VALUE = .25
# integral term
integral_reset = 0
# proportion
K = 1
# time steps per repeat
tau_i = 5
tau_d = 5
# last error
last_error = predictor.predictions(IMG_FILENAME.format(10)) - SET_VALUE

avg_errors = []
# Apply feedback controller for 20 more timesteps.
for j in range(11, 31):

    # calculate error
    error = predictor.predictions(IMG_FILENAME.format(j - 1)) - SET_VALUE

    # term to plot later
    avg_errors.append(sum(error) / 200.0)

    # update integral term
    integral_reset += error / tau_i

    # derivative term
    d_term = error - last_error

    # Update irrigation rate using feedback.
    vy.irrigation_rate += K * (error + integral_reset + d_term)

    # add noise to irrigation system output
    vy.irrigation_rate += np.random.normal(scale=STD_DEV, size=vy.irrigation_rate.shape)

    # Prevent irrigation rate from becoming negative.
    vy.irrigation_rate = vy.irrigation_rate.clip(min=0.0)

    # Log total amount of irrigation used.
    total_irrigation_used += np.sum(vy.irrigation_rate)

    # Run simulation for one timestep.
    vy.update(0)

    # Save image.
    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(IMG_FILENAME.format(j), bbox_inches=extent)
    size = 320, 320
    im = Image.open(IMG_FILENAME.format(j))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(IMG_FILENAME.format(j), "PNG")


print("Visualization Complete")
print("Drain Rate:", RATE, "Standard Dev:", STD_DEV)
print("File location:", IMG_FILENAME)
print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP:", total_irrigation_used / 200.0 / 10.0)
num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
print("Avg Number of Leaves per plant:", num_leaves / 200.0)
print("Irrigation per Leaf:", total_irrigation_used / num_leaves)
t = np.arange(10, 31)
print(t)
print(avg_errors)
error_plot = plt.figure()
error_plot.scatter(t, avg_errors)
error_plot.savefig(DIRECTORY + "error-plot")
