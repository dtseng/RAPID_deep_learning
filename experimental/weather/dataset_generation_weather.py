
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from PIL import Image

from matplotlib.animation import FuncAnimation
#from mpl_toolkits.axes_grid1 import make_axes_locatable

MIN_SIGMA = 10.0
MAX_SIGMA = 40.0

potted_plants_mode = False # each plant with independent drainages.  

def soil_variation(X):
    num_guassians = np.random.randint(2, 5)

    pd = np.zeros(200)

    for i in range(num_guassians):
        mu_x = np.random.uniform(100.0)
        mu_y = np.random.uniform(100.0)
        sigma_x = np.random.uniform(MIN_SIGMA, MAX_SIGMA)
        sigma_y = np.random.uniform(MIN_SIGMA, MAX_SIGMA)
        pd += np.exp(-((X[0] - mu_x)**2/( 2*sigma_x**2)) -
               ((X[1] - mu_y)**2/(2*sigma_y**2)))

    return np.clip(1.25 * pd, 0.0, 4.0)

class Plant(object):
    def __init__(self,position,init_soilmoisture):
        if potted_plants_mode:
            self.drainage_rate = np.random.uniform(0, 4) # for potted plants mode
        self.position=position
        self.soil_moisture=init_soilmoisture
        #self.drainage_rate=0
        #self.irrigation_rate=1
        #size of the leaves
        self.leaf_size=2 #size of each leaf for plotting
        self.leaf_num=7  #number of leaves to add each time the plant grows
        self.growth_ratio=2*np.array([[.05, 0], [0, 0.2]])
        self.leaf_positions = np.random.multivariate_normal(self.position, self.soil_moisture*self.growth_ratio, self.leaf_num)
        self.color()

    def grow(self, irrigation_rate, drainage_rate, evaporation):
        self.soil_moisture = self.soil_moisture + irrigation_rate - 0.5 * (drainage_rate + evaporation)
        if self.soil_moisture >= 0:
            newleaves = np.random.multivariate_normal(self.position, self.soil_moisture*self.growth_ratio, self.leaf_num)
            self.leaf_positions=np.vstack((self.leaf_positions,newleaves))
        else:
            self.soil_moisture=0
        self.color()
        

    def color(self):
        nlvs=len(self.leaf_positions[:, 0])

        if self.soil_moisture>0:
            colors= [0,153/256.0,0,.2]+np.random.uniform(-.2, .2, (nlvs,4))+[-10/256.0,0,0,0]
            # colors= np.zeros(colors.shape)
        else:
            colors=[128/256.0, 128/256.0, 0/256.0,1]+np.random.uniform(-.1, .1, (nlvs,4))
            # colors= np.zeros(colors.shape)

        colors=colors.tolist()
        self.colors= np.clip(colors, 0, 1)

class Vineyard(object):
    def __init__(self):
        # Create new Figure and an Axes which fills it.
        self.bounds=[[0,100],[0,100]]
        self.fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2,width_ratios=[1,1])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        #self.ax1=self.fig.add_subplot(121)
        #self.ax2=self.fig.add_subplot(122)
        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')
        #ax1 = fig.add_axes([0, 0, 1, 1], frameon=True)
        self.ax1.set_xlim(-10, self.bounds[0][1]+10), self.ax1.set_xticks([])
        self.ax1.set_ylim(-10, self.bounds[1][1]+10), self.ax1.set_yticks([])
        self.ax2.set_xlim(-10, self.bounds[0][1]+10), self.ax2.set_xticks([])
        self.ax2.set_ylim(-10, self.bounds[1][1]+10), self.ax2.set_yticks([])
        # self.ax1.set_axis_bgcolor((152/256.0, 107/256.0, 73/256.0))
        self.ax1.set_axis_bgcolor((145/256.0, 78/256.0, 25/256.0))


        # Initialize the plants on a grid
        nx, ny = (10, 20)
        x = np.linspace(0, self.bounds[0][1], nx)
        y = np.linspace(0, self.bounds[1][1], ny)
        xx, yy = np.meshgrid(x, y)

        self.vine_positions = np.vstack((xx.flatten(),yy.flatten())).T

        # print self.vine_positions.shape

        #each vine as it's own drainage rate
        self.drainage_rate = soil_variation(self.vine_positions.T)

        # Amount of rain to add at each timestep.
        self.evaporation = np.random.uniform(0.0, 4.0)

        # print self.drainage_rate
        # print self.drainage_rate.shape
        
        #each vine as it's own irrigation rate
        
        #constant irrigation
        self.irrigation_rate = 2 * np.ones(self.vine_positions.shape[0])
        
        #optimal irrigation
        # self.irrigation_rate =1*np.ones(self.vine_positions.shape[0])+self.drainage_rate
        
        #initialize the starting oil moisture level
        init_soilmoisture=1

        #create vines
        self.vines=[]
        for pos in self.vine_positions:
            self.vines.append(Plant(pos,init_soilmoisture))

        #set up some plotting stuff
        dr= self.drainage_rate.reshape(20,10)
        sc=self.ax2.imshow(np.flipud(dr),cmap=plt.get_cmap('RdBu_r'),extent=(0, self.bounds[0][1],0, self.bounds[1][1]))
        #legend
        df=self.fig.colorbar(sc,fraction=0.046, pad=0.04)
        df.ax.set_yticklabels(['slow','','','','','','','fast'])

        df.set_label('soil drainage', rotation=270)

        #plt.tight_layout()
        self.ind=0
        
    def update(self,i):
        # self.leaf_num=self.leaf_num+5
        #self.soil_moisture=self.soil_moisture-self.drainage_rate+self.irrigation_rate

        #leafpositions=[]
        sizes=[]
        moistures=[]
        for ind,vine in enumerate(self.vines):
            if potted_plants_mode:
                vine.grow(self.irrigation_rate[ind], vine.drainage_rate, self.evaporation)
            else:
                vine.grow(self.irrigation_rate[ind], self.drainage_rate[ind], self.evaporation)

            # plot stuff    
            if ind==0:

                leafpositions=vine.leaf_positions
                colors=vine.colors
                vinepositions=vine.position
            else:
                leafpositions=np.vstack((leafpositions,vine.leaf_positions))
                colors=np.vstack((colors,vine.colors))
                vinepositions=np.vstack((vinepositions,vine.position))

            sizes.append(vine.leaf_size)
            moistures.append(vine.soil_moisture)
        # print leafpositions.shape
        scat1 = self.ax1.scatter(leafpositions[:, 0], leafpositions[:, 1],
                            s=sizes,  edgecolors=colors,
                            facecolors=colors
                            )
        scat2 = self.ax2.scatter(vinepositions[:,0],vinepositions[:,1],
                            s=5*self.irrigation_rate,  edgecolors='red',
                            facecolors='red'
                            )

        self.ind=self.ind+1
        #pt.savefig('img'+str(self.ind)+'.png',dpi=1000)

    def animate(self):
        anim=FuncAnimation(self.fig, self.update)
        #anim.save('c:\optimal_irrigation.mp4', fps=20,bitrate=100000)
        # print 'done saving'
        plt.show()

NUM_TRAIN = 1000
NUM_VAL = 200
NUM_TEST = 200
NUM_UPDATES = 10
TRAIN_DR_FILE1 = "weather_data/train_data/reshaped/drain_rate{0}.npy"
TRAIN_DR_FILE2 = "weather_data/train_data/regular/drain_rate{0}.npy"
TRAIN_RAIN_FILE1 = "weather_data/train_data/reshaped/weather{0}.npy"
TRAIN_RAIN_FILE2 = "weather_data/train_data/regular/weather{0}.npy"
TRAIN_IMAGE_FILE1 = "weather_data/train_data/reshaped/image{0}.png"
TRAIN_IMAGE_FILE2 = "weather_data/train_data/regular/image{0}.png"

VAL_DR_FILE1 = "weather_data/val_data/reshaped/drain_rate{0}.npy"
VAL_DR_FILE2 = "weather_data/val_data/regular/drain_rate{0}.npy"
VAL_IMAGE_FILE1 = "weather_data/val_data/reshaped/image{0}.png"
VAL_IMAGE_FILE2 = "weather_data/val_data/regular/image{0}.png"
VAL_RAIN_FILE1 = "weather_data/val_data/reshaped/weather{0}.npy"
VAL_RAIN_FILE2 = "weather_data/val_data/regular/weather{0}.npy"

TEST_DR_FILE1 = "weather_data/test_data/reshaped/drain_rate{0}.npy"
TEST_DR_FILE2 = "weather_data/test_data/regular/drain_rate{0}.npy"
TEST_IMAGE_FILE1 = "weather_data/test_data/reshaped/image{0}.png"
TEST_IMAGE_FILE2 = "weather_data/test_data/regular/image{0}.png"
TEST_RAIN_FILE1 = "weather_data/test_data/reshaped/weather{0}.npy"
TEST_RAIN_FILE2 = "weather_data/test_data/regular/weather{0}.npy"

for i in range(NUM_TRAIN):
    vy = Vineyard()

    np.save(TRAIN_DR_FILE1.format(i), vy.drainage_rate / 4.0)
    np.save(TRAIN_DR_FILE2.format(i), vy.drainage_rate / 4.0)

    np.save(TRAIN_RAIN_FILE1.format(i), vy.evaporation / 4.0)
    np.save(TRAIN_RAIN_FILE1.format(i), vy.evaporation / 4.0)

    for _ in range(NUM_UPDATES):
        vy.update(0)

    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(TRAIN_IMAGE_FILE1.format(i), bbox_inches=extent)

    size = 420, 460
    im = Image.open(TRAIN_IMAGE_FILE1.format(i))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(TRAIN_IMAGE_FILE1.format(i), "PNG")  

    size = 320, 320
    im = Image.open(TRAIN_IMAGE_FILE1.format(i))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(TRAIN_IMAGE_FILE2.format(i), "PNG")

    plt.clf()
    plt.cla()
    plt.close()

# for i in range(NUM_VAL):
#     vy = Vineyard()

#     np.save(VAL_DR_FILE1.format(i), vy.drainage_rate / 4.0)
#     np.save(VAL_DR_FILE2.format(i), vy.drainage_rate / 4.0)

#     np.save(VAL_RAIN_FILE1.format(i), vy.evaporation / 4.0)
#     np.save(VAL_RAIN_FILE1.format(i), vy.evaporation / 4.0)

#     for _ in range(NUM_UPDATES):
#         vy.update(0)

#     extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
#     vy.fig.savefig(VAL_IMAGE_FILE1.format(i), bbox_inches=extent)

#     size = 420, 460
#     im = Image.open(VAL_IMAGE_FILE1.format(i))
#     im_resized = im.resize(size, Image.ANTIALIAS)
#     im_resized.save(VAL_IMAGE_FILE1.format(i), "PNG")  

#     size = 320, 320
#     im = Image.open(VAL_IMAGE_FILE1.format(i))
#     im_resized = im.resize(size, Image.ANTIALIAS)
#     im_resized.save(VAL_IMAGE_FILE2.format(i), "PNG")

#     plt.clf()
#     plt.cla()
#     plt.close()

for i in range(NUM_TEST):
    vy = Vineyard()

    np.save(TEST_DR_FILE1.format(i), vy.drainage_rate / 4.0)
    np.save(TEST_DR_FILE2.format(i), vy.drainage_rate / 4.0)

    np.save(TEST_RAIN_FILE1.format(i), vy.evaporation / 4.0)
    np.save(TEST_RAIN_FILE1.format(i), vy.evaporation / 4.0)

    for _ in range(NUM_UPDATES):
        vy.update(0)

    extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
    vy.fig.savefig(TEST_IMAGE_FILE1.format(i), bbox_inches=extent)

    size = 420, 460
    im = Image.open(TEST_IMAGE_FILE1.format(i))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(TEST_IMAGE_FILE1.format(i), "PNG")  

    size = 320, 320
    im = Image.open(TEST_IMAGE_FILE1.format(i))
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(TEST_IMAGE_FILE2.format(i), "PNG")

    plt.clf()
    plt.cla()
    plt.close()

# vy=Vineyard()
# vy.animate()
# animation = FuncAnimation(fig, vy.update,frames=5, interval=20)
# #animation.save('basic_animation2.mp4', fps=50,bitrate=5000)
# plt.show()
