# import the necessary packages
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths# store the image and mask filepaths, and augmentation
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		return len(self.imagePaths) # return the number of total samples contained in the dataset
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx] # grab the image path from the current index
		image = cv2.imread(imagePath) 	# load the image from disk
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #swap its channels from BGR to RGB,
		mask = cv2.imread(self.maskPaths[idx], 0) # read the associated mask from disk in grayscale mode
		if self.transforms is not None: 	# check to see if we are applying any transformations
			image = self.transforms(image) # apply the transformations to both image and its mask
			mask = self.transforms(mask)
		return (image, mask) # return a tuple of the image and its mask