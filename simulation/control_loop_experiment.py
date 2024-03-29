""" Experiment to determine amount of water saved using precision feedback control loop compared to flood irrigation. """

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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Computes amount of water used from timesteps 10-20 if we simply use the maximum predicted drainage rate as a reference,
# and apply the same amount of irrigation to all plants.
def flood_irrigation(predictor):
    total_irrigation_used = 0.0

    # Run experiment over all test set vineyards.
    for i in range(200):
        # Apply irrigation equal to 0.25 more than the maximum predicted drainage rate over all 10 timesteps.
        total_irrigation_used += 10.0 * 200.0 * (np.max(predictor.predictions("./datasets/noise_0/test_data/regular/image{0}.png".format(i))) + 0.25)

    print("TOTAL IRRIGATION (FLOOD) PER PLANT PER TIMESTEP:", total_irrigation_used / 200.0 / 200.0 / 10.0)


# Computes amount of water used from timesteps 10-20 using the precision feedback controller.
def precision_irrigation(predictor):
    total_irrigation_used = 0.0

    # Run experiment over all test set vineyards.
    for i in range(200):
        print("Running Precision Irrigation Experiment on Vineyard {0}.".format(i))

        vy = simulation.Vineyard()
        vy.drainage_rate = np.load("datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(i))

        # Update for 10 timesteps.
        for _ in range(10):
            vy.update(0)

        # Apply feedback controller for 10 timesteps.
        for _ in range(10):
            # Save image.
            extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
            vy.fig.savefig("test.png", bbox_inches=extent)
            size = 320, 320
            im = Image.open("test.png")
            im_resized = im.resize(size, Image.ANTIALIAS)
            im_resized.save("test.png", "PNG")

            # Apply feedback update equation.
            vy.irrigation_rate += predictor.predictions("test.png") - 0.25

            # Prevent irrigation rate from becoming negative.
            vy.irrigation_rate = vy.irrigation_rate.clip(min=0.0)

            # Log total amount of irrigation used.
            total_irrigation_used += np.sum(vy.irrigation_rate)

            # Run simulation for one timestep.
            vy.update(0)

        # Close figures.
        plt.clf()
        plt.cla()
        plt.close()

        print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP:", total_irrigation_used / (i + 1) / 200.0 / 10.0)

    print("TOTAL IRRIGATION (PRECISION) PER PLANT PER TIMESTEP:", total_irrigation_used / 200.0 / 200.0 / 10.0)


def main():
    predictor = predictions.Predictor("./saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())
    flood_irrigation(predictor)
    precision_irrigation(predictor)

if __name__ == '__main__':
    main()
