# I used: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# Annotation tool for Semantic Segmentation: label-studio start
# Another instance segmentation annotation tool: https://docs.hyperlabel.com/get-started-1/start-here
# Will use multi-class segmentation: https://medium.com/@mhamdaan/multi-class-semantic-segmentation-with-u-net-pytorch-ee81a66bba89
# https://github.com/wkentaro/labelme#windows
# I used this for box tracking instance segmentation:: https://app.plainsight.ai/label-and-train/datasets/01GEHT48ARH1F8RJK86KKFRKBB/labeler/01GEHT7TPN5MQJZ5XDM7XGC49R

# import the necessary packages
from errno import EIDRM
from time import process_time_ns
import torch
import os

DATASET_PATH = os.path.join("/media/ms/D/myGithub_Classified/Dataset_Segmentation/dataset/", "train") # base path of the dataset
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
NUM_EPOCHS = 10 # number of epochs to train for
BATCH_SIZE = 64 # batch size
INPUT_IMAGE_WIDTH = 128 # define the input image dimensions
INPUT_IMAGE_HEIGHT = 128
THRESHOLD = 0.5 # define threshold to filter weak predictions
BASE_OUTPUT = "output" # define the path to the base output directory
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth") # define the path to the output serialized model, model training
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"]) # plot, and testing image paths