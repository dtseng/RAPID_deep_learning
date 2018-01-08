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
        # Apply irrigation equal to the maximum predicted drainage rate over all 10 timesteps.
        total_irrigation_used += 10.0 * 200.0 * np.max(predictor.predictions("/home/wsong/datasets/noise_0/test_data/regular/image{0}.png".format(i)))

    print("TOTAL IRRIGATION PER PLANT PER TIMESTEP:", total_irrigation_used / 200.0 / 200.0 / 10.0)


# Computes amount of water used from timesteps 10-20 using the precision feedback controller.
def precision_irrigation(predictor):
    total_irrigation_used = 0.0

    # Run experiment over all test set vineyards.
    for i in range(200):
        if i % 5 == 0:
            print("Running Precision Irrigation Experiment on Vineyard {0}.".format(i))
        # "/home/wsong/datasets/noise_0/test_data/regular/drainage_rate{0}.npy"
        vy = simulation.Vineyard()
        vy.drainage_rate = np.load("/home/davidtseng/irrigation/datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(i))

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
            vy.irrgation_rate = vy.irrigation_rate.clip(min=0.0)

            # Log total amount of irrigation used.
            total_irrigation_used += np.sum(vy.irrigation_rate)

            # Run simulation for one timestep.
            vy.update(0)

    print("TOTAL IRRIGATION PER PLANT PER TIMESTEP:", total_irrigation_used / 200.0 / 200.0 / 10.0)


def main():
    # SAVED_MODEL_LOCATION = "/home/wsong/saved_models/whole_image/noise_0_training_1000.ckpt"
    SAVED_MODEL_LOCATION = "/home/davidtseng/irrigation/saved_models/whole_image/noise_0_training_1000.ckpt"
    predictor = predictions.Predictor(SAVED_MODEL_LOCATION, tf.Session())
    flood_irrigation(predictor)
    # precision_irrigation(predictor)

if __name__ == '__main__':
    main()
