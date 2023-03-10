# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# Modified by: Roshan Panahi, 03/10/2023
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import utils
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor 
from engine import train_one_epoch, evaluate
from PIL import Image
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class myDataset(object):
    def __init__(self, root, transforms):
        self.masks_dir =[]
        self.masks=[]
        self.imgs=[]
        self.root = root
        self.transforms = transforms

        for file in list(sorted(os.listdir(root))):

            if file[-4:]=="jpeg":
                self.imgs.append(file)
                self.masks_dir.append(file[:-5])

            if file[-3:]=="jpg":
                self.imgs.append(file)
                self.masks_dir.append(file[:-4])     
            
        for i in range (len(self.masks_dir)): # self.masks=[[masks for first image], [masks for second image], ...]
            files=[]

            for file in os.listdir(root+self.masks_dir[i]+"/"):
                files.append((root+self.masks_dir[i]+"/")+file)
            self.masks.append(files)

    def __getitem__(self, idx):  # load images and masks
   
        img_path = self.root+ self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        masks=[]
        boxes=[]
        my_class=[]
        np.set_printoptions(threshold=sys.maxsize)

        for file in range (len(self.masks[idx])): #TODO: Create a collage of RGB, and masked images for the paper. 
            mask=Image.open(self.masks[idx][file])  # self.masks[idx]: indicates the mask folder. file: the mask file per instance. 
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:] # obj_ids[] always has 2 elements. 
            my_class.append(obj_ids) 
            mask_binary = mask == obj_ids[0] # there is only one ID in each instance despite bg.
            masks.append(mask_binary)
            pos=np.where(mask_binary)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs=len(masks)
        labels=torch.squeeze(torch.tensor(np.array(my_class),dtype=torch.int64))
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.as_tensor([idx],dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # imagine box tensors in 3 dimension. 
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)     # load an instance segmentation model pre-trained pre-trained on COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features     # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    # replace the pre-trained head with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels     # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)     # and replace the mask predictor with a new one
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def myVisualize(mydataset): # TODO: should show 4 images, but only shows 1. Check again and fix. 
    #TODO: add  "Tracking model training with TensorBoard" to visulize the training process. 
    writer = SummaryWriter('runs/modular_experiment_1')
    dataiter = iter(mydataset) # get some random training images
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images) # create grid of images
    matplotlib_imshow(img_grid, one_channel=True) # show images
    writer.add_image('four_modular_images', img_grid)    # write to tensorboard

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device is:")
    print(device)
    num_classes = 3 #  dataset has three classes: {background, Module, BathPod}
    
    root="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/all_in_onePLace_set_1/"
    dataset = myDataset(root, get_transform(train=True))     # use our dataset and defined transformations
    root="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/all_in_one_place_augmentation_no/"
    dataset_test = myDataset(root, get_transform(train=False))
    #indices = torch.randperm(len(dataset)).tolist()    # split the dataset in train and test set --> Randomize

    indices = [*range(0,(len(dataset)))]   # split the dataset in train and test set --> Don't Randomize

    my_dataset = torch.utils.data.Subset(dataset, indices[0:1500])
    my_dataset_test = torch.utils.data.Subset(dataset_test, indices[31:43])

    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn) # define training and validation data loaders
    data_loader_test = torch.utils.data.DataLoader(
        my_dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
 
    model = get_model_instance_segmentation(num_classes)     # get the model using our helper function
    model.to(device)     # move model to the right device
    params = [p for p in model.parameters() if p.requires_grad]     # construct an optimizer
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)     # and a learning rate scheduler
    #myVisualize(dataset)
    num_epochs = 100   # let's train it for 10 epochs
    my_mAP_Seg=[]
    my_mAP_bbox=[]
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1) # train for one epoch, printing every 10 iterations
        lr_scheduler.step()         # update the learning rate
        img_write=False  # evaluate on the test dataset
        if (epoch==2):
            img_write=True

        print("evaluating===============================================")
        coco_evaluator=evaluate(model, data_loader_test,img_write, device=device)
        my_mAP_Seg.append('%.2f' % (coco_evaluator.coco_eval['segm'].stats[0]))
        if (epoch==5):
            plt.plot(range(0,epoch+1),my_mAP_Seg)
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.xlabel("Epoch")
            plt.ylabel("mAP")
            plt.show()

    print("That's it!!")

if __name__ == "__main__":
    main()