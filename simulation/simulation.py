import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from matplotlib.animation import FuncAnimation
from scipy.misc import imread


# ----------------- PARAMETERS FOR DATASET EXAMPLE GENERATION -----------------

# Range of number of areas with high dissipation rates.
MIN_NUM_GAUSSIANS = 2
MAX_NUM_GAUSSIANS = 4

# Range of standard deviation of multivariate Gaussians in x-axis direction.
MIN_SIGMA_X = 10.0
MAX_SIGMA_X = 40.0

# Range of standard deviation of multivariate Gaussians in y-axis direction.
MIN_SIGMA_Y = 10.0
MAX_SIGMA_Y = 40.0

# If true, each plant has independent dissipation uniformly at random between 0
# and 1. Otherwise, dataset is generated by summing Gaussian areas of high
# dissipation rate.
potted_plants_mode = False

# -----------------------------------------------------------------------------


# Takes a 2 x 200 matrix containing the positions of all plants in the field
# and generates a vector of soil moisture dissipation rates for each of the 200
# plants.
def soil_variation(X):
    # Choose the number of areas of high dissipation uniformly at random in the
    # specified range.
    num_gaussians = np.random.randint(MIN_NUM_GAUSSIANS, MAX_NUM_GAUSSIANS + 1)

    # Vector of soil moisture dissipation at each of the plants.
    dr = np.zeros(200)

    # Model each area of high dissipation rate as a scaled multivariate normal 
    # distribution.
    for _ in range(num_gaussians):
        # Choose the center uniformly at random in the field.
        mu_x = np.random.uniform(100.0)
        mu_y = np.random.uniform(100.0)

        # Choose the standard deviations in the x-axis and y-axis directions
        # uniformly at random in the specified ranges.
        sigma_x = np.random.uniform(MIN_SIGMA_X, MAX_SIGMA_X)
        sigma_y = np.random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y)

        # Add the scaled multivariate Gaussian distribution to the vector of
        # soil moisture dissipation rates.
        dr += 0.4 * (np.exp(-((X[0] - mu_x) ** 2) / (2 * (sigma_x ** 2))
              - ((X[1] - mu_y) ** 2) / (2 * (sigma_y ** 2))))

    # Normalize dissipation rates so that they are between 0 and 1.
    return np.clip(dr, 0.0, 1.0)

# Class for individual plants in the vineyard.
class Plant(object):
    def __init__(self, position, init_soil_moisture):
        # Set soil soil moisture dissipation rate of this plant independently
        # of other plants if in potted plants mode.
        if potted_plants_mode:
            self.dissipation_rate = np.random.uniform(0, 1)
        
        # Position of the plant in the field.
        self.position = position
        
        # Initialize local soil moisture.
        self.soil_moisture = init_soil_moisture
        
        # Size of the leaves for plotting.
        self.leaf_size = 1
        
        # Number of leaves to add each time the plant grows.
        self.leaf_num = 7
        
        # When growing, leaf positions are sampled as a multivariate Gaussian
        # distribution with mean equal to the position of the plant and
        # covariance matrix proportional to this growth ratio.
        self.growth_ratio = np.array([[0.3, 0], [0, 1.2]])
        
        # Sample the initial leaf positions of the plants.
        self.leaf_positions = np.random.multivariate_normal(self.position, 
                                                            self.soil_moisture * self.growth_ratio, 
                                                            self.leaf_num)
        
        # Assign colors to each leaf for the visualization.
        self.color()

    # Simulate plant growth using the supplied irrigation rate and soil
    # moisture dissipation rate.
    def grow(self, irrigation_rate, dissipation_rate, noise):
        # First-order approximation of Richards equation.
        self.soil_moisture += irrigation_rate - dissipation_rate + np.random.normal(scale=noise)

        # New leaves are added if soil moisture is non-negative.
        if self.soil_moisture>=0:
            new_leaves = np.random.multivariate_normal(self.position, 
                                                       self.soil_moisture * self.growth_ratio, 
                                                       self.leaf_num)
            self.leaf_positions=np.vstack((self.leaf_positions, new_leaves))
        # Soil moisture cannot be negative.
        else:
            self.soil_moisture = 0

        # Color the new leaves.
        self.color()

    # Assign colors to each of the leaves for the plant.
    def color(self):
        num_leaves = len(self.leaf_positions[:, 0])

        # We use shades of green for the leaves if soil moisture is positive.
        if self.soil_moisture>0:
            colors = [0, 153/256.0, 0, 0.2] + np.random.uniform(-.2, .2, (num_leaves, 4)) + [-10/256.0, 0, 0, 0]
        # We use shades of yellow if the soil moisture is zero.
        else:
            colors = [128/256.0, 128/256.0, 0/256.0, 1] + np.random.uniform(-.1, .1, (num_leaves, 4))

        colors = colors.tolist()

        # Ensure that colors are between 0 and 1.
        self.colors= np.clip(colors, 0, 1)

# Class for a vineyard object, which consists of individual plants.
class Vineyard(object):
    def __init__(self):
        # Create new Figure and an Axes which fills it.
        self.bounds = [[0,100],[0,100]]
        self.fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2,width_ratios=[1,1])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')
        self.ax1.set_xlim(-10, self.bounds[0][1]+10), self.ax1.set_xticks([])
        self.ax1.set_ylim(-10, self.bounds[1][1]+10), self.ax1.set_yticks([])
        self.ax2.set_xlim(-10, self.bounds[0][1]+10), self.ax2.set_xticks([])
        self.ax2.set_ylim(-10, self.bounds[1][1]+10), self.ax2.set_yticks([])
        self.ax1.imshow(imread("soil.jpg"), extent=[-10, 110, -10, 110])

        # Initialize the plants on a grid
        nx, ny = (10, 20)
        x = np.linspace(0, self.bounds[0][1], nx)
        y = np.linspace(0, self.bounds[1][1], ny)
        xx, yy = np.meshgrid(x, y)

        # Positions of the individual plants in the field.
        self.vine_positions = np.vstack((xx.flatten(),yy.flatten())).T

        # Set local soil moisture dissipation rate for each individual plant.
        self.dissipation_rate = soil_variation(self.vine_positions.T)
        
        # Irrigation rate vector (one irrigation rate per plant).
        self.irrigation_rate = 0.5 * np.ones(self.vine_positions.shape[0])
        
        # Starting soil moisture level for each plant.
        init_soil_moisture = 1

        # Create each plant.
        self.vines = []
        for pos in self.vine_positions:
            self.vines.append(Plant(pos, init_soil_moisture))

        # Set up plotting for visualization.
        dr = self.dissipation_rate.reshape(20,10)
        sc = self.ax2.imshow(np.flipud(dr), cmap=plt.get_cmap('RdBu_r'), 
                             extent=(0, self.bounds[0][1], 0, self.bounds[1][1]),
                             interpolation='gaussian')
        
        # Legend.
        df=self.fig.colorbar(sc,fraction=0.046, pad=0.04)
        df.ax.set_yticklabels(['slow','','','','','', 'fast'])
        
        # Keep track of the current day.
        self.time = 0
        
    # Simulates plant growth over one day.
    def update(self, i, noise=0):
        sizes = []
        moistures = []

        for ind,vine in enumerate(self.vines):
            # Simulate plant growth.
            if potted_plants_mode:
                vine.grow(self.irrigation_rate[ind], vine.dissipation_rate, noise)
            else:
                vine.grow(self.irrigation_rate[ind], self.dissipation_rate[ind], noise)

            # Get information for generating synthetic aerial image.   
            if ind == 0:
                leafpositions = vine.leaf_positions
                colors = vine.colors
                vinepositions = vine.position
            else:
                leafpositions = np.vstack((leafpositions,vine.leaf_positions))
                colors = np.vstack((colors,vine.colors))
                vinepositions = np.vstack((vinepositions,vine.position))

            sizes.append(vine.leaf_size)
            moistures.append(vine.soil_moisture)

        # Create synthetic aerial image.
        scat1 = self.ax1.scatter(leafpositions[:, 0], leafpositions[:, 1],
                                 s=sizes, edgecolors=colors,
                                 facecolors=colors)
        scat2 = self.ax2.scatter(vinepositions[:,0],vinepositions[:,1],
                                 s=5*self.irrigation_rate, edgecolors='red',
                                 facecolors='red')

        # Increment the timestep.
        self.time += 1

    # Create an animation showing plant growth over time.
    def animate(self):
        anim = FuncAnimation(self.fig, self.update)
        plt.show()
