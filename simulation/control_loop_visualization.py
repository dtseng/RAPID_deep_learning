""" Script for saving images of the vineyard over time after applying the control loop. """

import simulation
import predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from PIL import Image
from matplotlib.animation import FuncAnimation

# Initialize example vineyard.
vy = simulation.Vineyard()
vy.drainage_rate = np.load("./datasets/noise_0/test_data/regular/drainage_rate0.npy".format(i))

# Update vineyard for 10 timesteps.
for i in range(10):
    vy.update(0)

# File pattern for save images.
IMG_FILENAME = "test/test_img{0}.png"

# Save current image of vineyard.
extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
vy.fig.savefig(IMG_FILENAME.format(10), bbox_inches=extent)
size = 320, 320
im = Image.open(IMG_FILENAME.format(10))
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save(IMG_FILENAME.format(10), "PNG")

# Apply feedback controller for 20 more timesteps.
for j in range(11, 31):
    # Update irrigation rate using feedback.
    vy.irrigation_rate += predictions.predictions(IMG_FILENAME.format(j - 1)) - 0.25

    # Run simulation for one timestep.
    vy.update(0)

    # Save image.
    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(IMG_FILENAME.format(j), bbox_inches=extent)
    size = 320, 320
    im = Image.open(IMG_FILENAME.format(j))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(IMG_FILENAME.format(j), "PNG")
