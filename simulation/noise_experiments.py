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
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ADJUST_SCALE = .1
MECH_SCALE = .1
SPATIAL_RATE = .05
VINEYARD_SHAPE = (20,10)
NUM_TRIALS = 20
COORDS = {0: (1,0), 1: (0, -1), 2: (-1, 0), 3: (0,1)}
def add_spatial_noise(rates):
    original_shape = rates.shape
    rates = rates.reshape(VINEYARD_SHAPE)
    # sample which emitters will be switched
    p = (1 - SPATIAL_RATE, SPATIAL_RATE)
    # size = (VINEYARD_SHAPE[0] - 2,VINEYARD_SHAPE[1] - 2)
    mask = np.random.choice((0,1), size=VINEYARD_SHAPE, p=p)
    print("number of swaps:", np.sum(mask[1:-1, 1:-1]))
    #decide which direction to switch, do not iterate through edges
    for i in range(1, VINEYARD_SHAPE[0]-1):
        for j in range(1, VINEYARD_SHAPE[1]-1):
            if mask[i][j] == 1:
                #randomly choose neighbor and swap
                coord = COORDS[np.random.choice(3)]
                temp = rates[i][j]
                rates[i][j] = rates[i+coord[0]][j+coord[1]]
                rates[i+coord[0]][j+coord[1]] = temp
    return rates.reshape(original_shape)


# Computes amount of water used from timesteps 10-20 if we simply use the maximum predicted drainage rate as a reference,
# and apply the same amount of irrigation to all plants.
def flood_irrigation(predictor, noise=None):
    total_irrigation_used = 0.0
    variances = []
    # Run experiment over all test set vineyards.
    for i in range(NUM_TRIALS):
        # Apply irrigation equal to 0.25 more than the maximum predicted drainage rate over all 10 timesteps.
        print("Running Flood Irrigation Experiment on Vineyard {0}.".format(i))

        vy = simulation.Vineyard()
        vy.drainage_rate = np.load("datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(i))

        # Update for 10 timesteps.
        for _ in range(10):
            vy.update(0)


        # save image to make prediciton off of
        extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
        vy.fig.savefig("test.png", bbox_inches=extent)
        size = 320, 320
        im = Image.open("test.png")
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save("test.png", "PNG")

        # make predictions
        rates = np.max(predictor.predictions("test.png")) + 0.25
        
        # add noise if neccesary
        if noise == "adjustments":
            rates += np.random.normal(scale=ADJUST_SCALE, size=fixed_rates.shape)
        
        vy.irrigation_rate = rates

        # grow with flood irrigation
        for _ in range(10):
            vy.update(0)

        # Close figures.
        plt.clf()
        plt.cla()
        plt.close()

        total_irrigation_used += 10.0 * 200.0 * rates

        moistures = np.array([p.soil_moisture for p in vy.vines])
        variances.append(np.var(moistures))
        num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
        avg_leaves = num_leaves / 200.0
        avg_irr_per_leaf = total_irrigation_used / num_leaves
        print("AVERAGE IRRIGATION PER LEAF:", avg_irr_per_leaf)

    print("TOTAL IRRIGATION (FLOOD) PER PLANT PER TIMESTEP NOISE = {}:".format(noise), total_irrigation_used / 200.0 / 10.0/ NUM_TRIALS)
    print("VARIANCE", variances)


# Computes amount of water used from timesteps 10-20 using the precision feedback controller.
def precision_irrigation(predictor, noise=None):
    total_irrigation_used = 0.0
    variances = []
    # Run experiment over all test set vineyards.
    for i in range(NUM_TRIALS):
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

            if noise == "adjustments":
                vy.irrigation_rate += np.random.normal(scale=ADJUST_SCALE, size=fixed_rates.shape)
            
            if noise == "spatial":
                vy.irrigation_rate = add_spatial_noise(vy.irrigation_rate) 
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

        print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP NOISE = {}:".format(noise), total_irrigation_used / (i + 1) / 200.0 / 10.0)
        moistures = np.array([p.soil_moisture for p in vy.vines])
        variances.append(np.var(moistures))
        num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
        avg_leaves = num_leaves / 200.0
        avg_irr_per_leaf = total_irrigation_used / num_leaves
        print("AVERAGE IRRIGATION PER LEAF:", avg_irr_per_leaf)

    print("TOTAL IRRIGATION (PRECISION) PER PLANT PER TIMESTEP NOISE = {}:".format(noise), total_irrigation_used / 200.0 / 10.0/ NUM_TRIALS)
    print("VARIANCE", variances)


def fixed_prediction_irrigation(predictor, noise=None):
    # Run experiment over all test set vineyards.
    variances = []
    for i in range(NUM_TRIALS):
        print("Running Fixed Prediction Irrigation Experiment on Vineyard {0}.".format(i))

        vy = simulation.Vineyard()
        vy.drainage_rate = np.load("datasets/noise_0/test_data/regular/drainage_rate{0}.npy".format(i))

        # Update for 10 timesteps.
        for _ in range(10):
            vy.update(0)

        extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
        vy.fig.savefig("test.png", bbox_inches=extent)
        size = 320, 320
        im = Image.open("test.png")
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save("test.png", "PNG")
        
        # predict fixed rates
        fixed_rates = predictor.predictions("test.png") + 0.25
        # Apply feedback controller for 10 timesteps.
        if noise == "spatial":
            fixed_rates = add_spatial_noise(fixed_rates)
        # add guassian noise to simulate human error of adjustments
        if noise == "adjustments":
            fixed_rates += np.random.normal(scale=ADJUST_SCALE, size=fixed_rates.shape)
            fixed_rates = fixed_rates.clip(min=0.0)
        vy.irrigation_rate = fixed_rates
        for _ in range(10):
            # Log total amount of irrigation used. Could potentially sum at end
            total_irrigation_used += np.sum(vy.irrigation_rate)

            # Run simulation for one timestep.
            vy.update(0)

        # Close figures.
        plt.clf()
        plt.cla()
        plt.close()

        print("AVERAGE IRRIGATION PER PLANT PER TIMESTEP NOISE = {}:".format(noise), total_irrigation_used / (i + 1) / 200.0 / 10.0)
        moistures = np.array([p.soil_moisture for p in vy.vines])
        variances.append(np.var(moistures))
        num_leaves = sum([len(p.leaf_positions) for p in vy.vines])
        avg_leaves = num_leaves / 200.0
        avg_irr_per_leaf = total_irrigation_used / num_leaves
        print("AVERAGE IRRIGATION PER LEAF:", avg_irr_per_leaf)

    print("TOTAL IRRIGATION (PRECISION) PER PLANT PER TIMESTEP NOISE = {}:".format(noise), total_irrigation_used / 200.0 / 10.0 / NUM_TRIALS)
    print("VARIANCE", variances)


def main():
    predictor = predictions.Predictor("./saved_models/whole_image/noise_0_training_1000.ckpt", tf.Session())
    flood_irrigation(predictor)
    precision_irrigation(predictor)
    fixed_prediction_irrigation(predictor)
    
    print("GUASSIAN ADJUSTMENT NOISE", "-"*40)
    flood_irrigation(predictor, noise="adjustments")
    precision_irrigation(predictor, noise="adjustments")
    fixed_prediction_irrigation(predictor, noise="adjustments")

    print("SPATIAL ADJUSTMENT NOISE", "-"*40)
    precision_irrigation(predictor, noise="spatial")
    fixed_prediction_irrigation(predictor, noise="spatial")

if __name__ == '__main__':
    main()