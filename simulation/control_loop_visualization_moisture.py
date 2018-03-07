""" Script for saving images of the vineyard over time after applying the control loop. """

import simulation
import predictions
import matplotlib as mpl
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
RATE = 0

#unshaped 200 item array to be plottec
def save_heat_map(x, title):
    x = np.reshape(x, (20,10))
    # print(title)
    # print(x)
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['red','white','blue'],
                                           256)
    img2 = plt.imshow(x, interpolation='nearest',
                    cmap = cmap2,
                    origin='lower',
                    extent=(0, 100, 0, 100))
    plt.savefig(title, bbox_inches='tight')


def save_actual_moisture_map(vy, title):
    x = []
    for plant in vy.vines:
        x.append(plant.soil_moisture)
    save_heat_map(x, title)


# Initialize example vineyard.
vy = simulation.Vineyard()
vy.drainage_rate = np.load("/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(RATE))

# File pattern for save images.
DIRECTORY = "test_drain_rate{0}_std_{1}_moisture/".format(RATE, STD_DEV)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

# Save image of disspation rates
extent = vy.ax2.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
FILE_NAME = DIRECTORY + "disspation_map.png"
vy.fig.savefig(FILE_NAME, bbox_inches=extent)
size = 320, 320
im = Image.open(FILE_NAME)
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save(FILE_NAME, "PNG")

# Update vineyard for 10 timesteps.
for i in range(10):
    vy.update(0)

# Save current image of vineyard.
IMG_FILENAME = DIRECTORY + "test_img{0}.png"
extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
vy.fig.savefig(IMG_FILENAME.format(10), bbox_inches=extent)
size = 320, 320
im = Image.open(IMG_FILENAME.format(10))
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save(IMG_FILENAME.format(10), "PNG")

# Intialize predictor.
predictor = predictions.Predictor("/home/wsong/saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())
# vy.irrigation_rate = predictor.predictions(IMG_FILENAME.format(10))
saved_rates = predictor.predictions(IMG_FILENAME.format(10))
print("saved_rates:", saved_rates)
print("saved_rates shape:", saved_rates.shape)

total_irrigation_used = 0.0
avg_errors = []
MOISTURE_SET_POINT = 1
curr_moisture = 1 + (vy.irrigation_rate - saved_rates)*10

#save initial moisture estimate/actual
print("initial est moisture", curr_moisture)
print("moisture shape", curr_moisture.shape)
MOISTURE_EST_NAME = DIRECTORY + "moisture_est_img{0}.png"
save_heat_map(curr_moisture, MOISTURE_EST_NAME.format(10))
MOISTURE_ACTUAL_NAME = DIRECTORY + "moisture_actual_img{0}.png"
save_actual_moisture_map(vy, MOISTURE_ACTUAL_NAME.format(10))

# Apply constant irrigation for 20 more timesteps.
for j in range(11, 31):

    rate_preds = predictor.predictions(IMG_FILENAME.format(j - 1))
    irrigation_error = rate_preds - 0.25
    
    # estimated distance of moisture from set point
    moisture_difference = MOISTURE_SET_POINT - (curr_moisture - rate_preds + vy.irrigation_rate)

    # errors to plot later
    avg_errors.append(sum(irrigation_error) / 200.0)

    # Update irrigation rate using feedback.
    vy.irrigation_rate += moisture_difference

    # add noise to irrigation system output
    # vy.irrigation_rate = saved_rates # + np.random.normal(scale=STD_DEV, size=vy.irrigation_rate.shape) + .25

    # Prevent irrigation rate from becoming negative.
    vy.irrigation_rate = vy.irrigation_rate.clip(min=0.0)

    # Log total amount of irrigation used.
    total_irrigation_used += np.sum(vy.irrigation_rate)

    # update soil moisture estimate
    curr_moisture += vy.irrigation_rate - rate_preds
    curr_moisture.clip(min=0.0)

    # Run simulation for one timestep.
    vy.update(0)

    # Save image of vines
    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(IMG_FILENAME.format(j), bbox_inches=extent)
    size = 320, 320
    im = Image.open(IMG_FILENAME.format(j))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(IMG_FILENAME.format(j), "PNG")

print("Visualization Complete")
print("Drain Rate:", RATE, "Standard Dev:", STD_DEV)
print("File location:", IMG_FILENAME)
avg_irrigation = total_irrigation_used / 200.0 / 20.0
num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
avg_leaves = num_leaves / 200.0
avg_irr_per_leaf = total_irrigation_used / num_leaves
print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP:", avg_irrigation)

print("Avg Number of Leaves per plant:", avg_leaves)
print("Irrigation per Leaf:", avg_irr_per_leaf)

title = "Avg leaves/plant: {0:.3f}, avg irrigation/plant/time step: {1:.3f}, irrigation/leaf: {2:.3f}".format(avg_leaves, avg_irrigation, avg_irr_per_leaf)
t = np.arange(11, 31)
error_plot = plt.figure()
ax = error_plot.add_subplot(111)
ax.scatter(t, avg_errors)
ax.set_title(title, fontsize=10, fontweight='bold')
error_plot.savefig(DIRECTORY + "error-plot")
