# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

# import the necessary packages
import torch
import os

DATASET_PATH = os.path.join("./", "data_train") # base path of the dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")# define the path to the images and masks dataset
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
TEST_SPLIT = 0.15 # define the test split
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # determine the device to be used for training and evaluation
PIN_MEMORY = True if DEVICE == "cuda" else False # determine if we will be pinning memory during data loading

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.001 #initialize learning rate,
NUM_EPOCHS = 40 # number of epochs to train for
BATCH_SIZE = 64 # batch size
INPUT_IMAGE_WIDTH = 128 # define the input image dimensions
INPUT_IMAGE_HEIGHT = 128
THRESHOLD = 0.5 # define threshold to filter weak predictions
BASE_OUTPUT = "output" # define the path to the base output directory
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth") # define the path to the output serialized model, model training
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"]) # plot, and testing image paths

