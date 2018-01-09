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
STD_DEV = 1
# drainage rate
RATE = 5

# Initialize example vineyard.
vy = simulation.Vineyard()
vy.drainage_rate = np.load("/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(RATE))

# Update vineyard for 10 timesteps.
for i in range(10):
    vy.update(0)

# File pattern for save images.
IMG_FILENAME = "test_drain_rate{0}_std_{1}/".format(RATE, STD_DEV)
IMG_FILENAME += "test_img{0}.png"

# Save current image of vineyard.
extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
vy.fig.savefig(IMG_FILENAME.format(10), bbox_inches=extent)
size = 320, 320
im = Image.open(IMG_FILENAME.format(10))
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save(IMG_FILENAME.format(10), "PNG")

# Intialize predictor.
predictor = predictions.Predictor("/home/wsong/saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())

# Apply feedback controller for 20 more timesteps.
for j in range(11, 31):
    # Update irrigation rate using feedback.
    vy.irrigation_rate += predictor.predictions(IMG_FILENAME.format(j - 1)) - 0.25

    # add noise to irrigation system output
    vy.irrigation_rate += np.random.normal(scale=STD_DEV, size=vy.irrigation_rate.shape)

    # Prevent irrigation rate from becoming negative.
    vy.irrigation_rate = vy.irrigation_rate.clip(min=0.0)

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

