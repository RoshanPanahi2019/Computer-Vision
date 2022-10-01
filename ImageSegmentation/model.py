# import the necessary packages
import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = Conv2d(inChannels, outChannels, 3) # store the convolution and RELU layers
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		return self.conv2(self.relu(self.conv1(x))) # apply CONV => RELU => CONV block to the inputs and return it

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		self.encBlocks = ModuleList( 
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2) # store the encoder blocks and maxpooling layer
	def forward(self, x):
		blockOutputs = [] # initialize an empty list to store the intermediate outputs
	
		for block in self.encBlocks: 	# loop through the encoder blocks
			x = block(x) # pass the inputs through the current encoder block, 
			blockOutputs.append(x) # store the outputs, 
			x = self.pool(x) #  and then apply maxpooling on the output
		return blockOutputs # return the list containing the intermediate outputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		self.channels = channels # initialize the number of channels, upsampler blocks, and
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)]) # decoder blocks
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		for i in range(len(self.channels) - 1): # loop through the number of channels
			x = self.upconvs[i](x) 		# pass the inputs through the upsampler blocks
			encFeat = self.crop(encFeatures[i], x) # crop the current features from the encoder blocks,
			x = torch.cat([x, encFeat], dim=1) # concatenate them with the current upsampled features,
			x = self.dec_blocks[i](x) # and pass the concatenated output through the current decoder block
		return x # return the final decoder output
	def crop(self, encFeatures, x):
		(_, _, H, W) = x.shape 	# grab the dimensions of the inputs, and crop the encoder
		encFeatures = CenterCrop([H, W])(encFeatures) # features to match the dimensions
		return encFeatures # return the cropped features

class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64),
        decChannels=(64, 32, 16),
        nbClasses=1, retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(encChannels) # initialize the encoder and decoder
        self.decoder = Decoder(decChannels)
        self.head = Conv2d(decChannels[-1], nbClasses, 1) # initialize the regression head and store the class variables
        self.retainDim = retainDim
        self.outSize = outSize

def forward(self, x):
    
    encFeatures = self.encoder(x) # grab the features from the encoder
    decFeatures = self.decoder(encFeatures[::-1][0],
        encFeatures[::-1][1:])   # pass the encoder features through decoder making sure that their dimensions are suited for concatenation
    map = self.head(decFeatures)  # pass the decoder features through the regression head to obtain the segmentation mask
    if self.retainDim: # check to see if we are retaining the original output
        map = F.interpolate(map, self.outSize)  # dimensions and if so, then resize the output to match them
    return map # return the segmentation map
    