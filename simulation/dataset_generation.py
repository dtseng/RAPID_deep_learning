""" Script for generating datasets of varying levels of simulation noise. """

import simulation

NUM_TRAIN = 2000    # Number of training examples.
NUM_VAL = 200       # Number of validation examples.
NUM_TEST = 200      # Number of test examples.
NUM_UPDATES = 10    # Number of time steps to run simulation for before saving the image.

# Various file patterns.
TRAIN_DR_FILE1 = "./datasets/noise_{0}/train_data/reshaped/drainage_rate{1}.npy"
TRAIN_DR_FILE2 = "./datasets/noise_{0}/train_data/regular/drainage_rate{1}.npy"
TRAIN_IMAGE_FILE1 = "./datasets/noise_{0}/train_data/reshaped/image{1}.png"
TRAIN_IMAGE_FILE2 = "./datasets/noise_{0}/train_data/regular/image{1}.png"
VAL_DR_FILE1 = "./datasets/noise_{0}/val_data/reshaped/drainage_rate{1}.npy"
VAL_DR_FILE2 = "./datasets/noise_{0}/val_data/regular/drainage_rate{1}.npy"
VAL_IMAGE_FILE1 = "./datasets/noise_{0}/val_data/reshaped/image{1}.png"
VAL_IMAGE_FILE2 = "./datasets/noise_{0}/val_data/regular/image{1}.png"
TEST_DR_FILE1 = "./datasets/noise_{0}/test_data/reshaped/drainage_rate{1}.npy"
TEST_DR_FILE2 = "./datasets/noise_{0}/test_data/regular/drainage_rate{1}.npy"
TEST_IMAGE_FILE1 = "./datasets/noise_{0}/test_data/reshaped/image{1}.png"
TEST_IMAGE_FILE2 = "./datasets/noise_{0}/test_data/regular/image{1}.png"

for noise in [0, 25, 50, 75, 100]:
    for i in range(NUM_TRAIN):
        vy = simulation.Vineyard()

        np.save(TRAIN_DR_FILE1.format(noise, i), vy.drainage_rate)
        np.save(TRAIN_DR_FILE2.format(noise, i), vy.drainage_rate)

        for _ in range(NUM_UPDATES):
            vy.update(noise / 100.0)

        extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
        vy.fig.savefig(TRAIN_IMAGE_FILE1.format(noise, i), bbox_inches=extent)

        size = 420, 460
        im = Image.open(TRAIN_IMAGE_FILE1.format(noise, i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(TRAIN_IMAGE_FILE1.format(noise, i), "PNG")  

        size = 320, 320
        im = Image.open(TRAIN_IMAGE_FILE1.format(noise, i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(TRAIN_IMAGE_FILE2.format(noise, i), "PNG")

        plt.clf()
        plt.cla()
        plt.close()


    for i in range(NUM_VAL):
        vy = Vineyard()

        np.save(VAL_DR_FILE1.format(i), vy.drainage_rate)
        np.save(VAL_DR_FILE2.format(i), vy.drainage_rate)

        for _ in range(NUM_UPDATES):
            vy.update(noise / 100.0)

        extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
        vy.fig.savefig(VAL_IMAGE_FILE1.format(i), bbox_inches=extent)

        size = 420, 460
        im = Image.open(VAL_IMAGE_FILE1.format(i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(VAL_IMAGE_FILE1.format(i), "PNG")  

        size = 320, 320
        im = Image.open(VAL_IMAGE_FILE1.format(i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(VAL_IMAGE_FILE2.format(i), "PNG")

        plt.clf()
        plt.cla()
        plt.close()

    for i in range(NUM_TEST):
        vy = simulation.Vineyard()

        np.save(TEST_DR_FILE1.format(noise, i), vy.drainage_rate)
        np.save(TEST_DR_FILE2.format(noise, i), vy.drainage_rate)

        for _ in range(NUM_UPDATES):
            vy.update(noise / 100.0)

        extent = vy.ax1.get_window_extent().transformed(vy.fig.dpi_scale_trans.inverted())
        vy.fig.savefig(TEST_IMAGE_FILE1.format(noise, i), bbox_inches=extent)

        size = 420, 460
        im = Image.open(TEST_IMAGE_FILE1.format(noise, i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(TEST_IMAGE_FILE1.format(noise, i), "PNG")  

        size = 320, 320
        im = Image.open(TEST_IMAGE_FILE1.format(noise, i))
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(TEST_IMAGE_FILE2.format(noise, i), "PNG")

        plt.clf()
        plt.cla()
        plt.close()
