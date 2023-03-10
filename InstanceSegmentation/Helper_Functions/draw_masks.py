import os
from PIL import Image
import torch
import numpy as np
import imageio
import cv2

mask_dir = "/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/all_in_one_place_set_2/24-0-0/"# Set the path to the directory containing the masks
mask_files = os.listdir(mask_dir)# Get a list of all the mask file names in the directory
masks=[]

i=0
width = 2592
height = 1944
combined_image= np.zeros((height, width), dtype=np.uint8)

for file in mask_files:
    if file.endswith(".png"):
        masks.append(cv2.imread(mask_dir+file, cv2.IMREAD_GRAYSCALE))
        combined_image = np.logical_or(combined_image, masks[i]).astype(np.uint8) * 255
        i=i+1

cv2.imwrite("combined_image.png", combined_image)

