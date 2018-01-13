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


def pid_irrigation():
    print("RUNNING ONE PID IRRIGATION")
    # control loop noise
    STD_DEV = 0
    # drainage rate
    RATE = 5
    # File pattern for save images.
    DIRECTORY = "pid_test_drain_rate{0}_std_{1}/".format(RATE, STD_DEV)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    IMG_FILENAME = DIRECTORY + "test_img{0}.png"


    # Initialize example vineyard.
    vy = simulation.Vineyard()
    vy.drainage_rate = np.load("/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(RATE))

    # Update vineyard for 10 timesteps.
    for i in range(10):
        vy.update(0)

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
    K = 10
    # time steps per repeat
    tau_i = 10
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
    avg_irrigation = total_irrigation_used / 200.0 / 20.0
    num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
    avg_leaves = num_leaves / 200.0
    avg_irr_per_leaf = total_irrigation_used / num_leaves
    print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP:", avg_irrigation)

    print("Avg Number of Leaves per plant:", avg_leaves)
    print("Irrigation per Leaf:", avg_irr_per_leaf)

    # plot error chart for PID tuning
    title = "Drain Rate: {0}, Standard Dev: {1}, Proportion (K): {2}, tau_i: {3}, tau_d: {4}".format(RATE, STD_DEV, K, tau_i, tau_d)
    suptitle = "Avg leaves per plant: {0:.3f}, avg irrigation per plant per time step: {1:.3f}, irrigation per leaf: {2:.3f}".format(avg_leaves, avg_irrigation, avg_irr_per_leaf)
    t = np.arange(11, 31)
    error_plot = plt.figure()
    ax = error_plot.add_subplot(111)
    ax.scatter(t, avg_errors)
    ax.set_title(title)
    error_plot.suptitle(suptitle, fontsize=10, fontweight='bold')
    error_plot.savefig(DIRECTORY + "error-plot")


def one_prediction_irrigation():
    print("RUNNING ONE PREDICTION IRRIGATION")
    # control loop noise
    STD_DEV = 0
    # drainage rate
    RATE = 5
    # File pattern for save images.
    DIRECTORY = "pid_test_drain_rate{0}_std_{1}/".format(RATE, STD_DEV)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    IMG_FILENAME = DIRECTORY + "test_img{0}.png"

    
    # Initialize example vineyard.
    vy = simulation.Vineyard()
    vy.drainage_rate = np.load("/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(RATE))

    # Update vineyard for 10 timesteps.
    for i in range(10):
        vy.update(0)

    # Save current image of vineyard.
    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(IMG_FILENAME.format(10), bbox_inches=extent)
    size = 320, 320
    im = Image.open(IMG_FILENAME.format(10))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(IMG_FILENAME.format(10), "PNG")

    SET_VALUE = .25

    # Intialize predictor.
    predictor = predictions.Predictor("/home/wsong/saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())
    predicted_rates = predictor.predictions(IMG_FILENAME.format(10))
    vy.irrigation_rate = predicted_rates + SET_VALUE

    total_irrigation_used = 0.0

    avg_errors = []
    # Apply feedback controller for 20 more timesteps.
    for j in range(11, 31):

        # calculate error
        error = predictor.predictions(IMG_FILENAME.format(j - 1)) - SET_VALUE

        # term to plot later
        avg_errors.append(sum(error) / 200.0)

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

        # reset to old irrigation rate
        vy.irrigation_rate = predicted_rates + SET_VALUE

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

    # plot error chart for PID tuning
    title = "Avg leaves per plant: {0:.3f}, avg irrigation per plant per time step: {1:.3f}, irrigation per leaf: {2:.3f}".format(avg_leaves, avg_irrigation, avg_irr_per_leaf)
    t = np.arange(11, 31)
    error_plot = plt.figure()
    ax = error_plot.add_subplot(111)
    ax.scatter(t, avg_errors)
    ax.set_title(title, fontsize=9, fontweight='bold')
    error_plot.savefig(DIRECTORY + "error-plot")


def main():
    # pid_irrigation()

    one_prediction_irrigation()


if __name__ == '__main__':
    main()