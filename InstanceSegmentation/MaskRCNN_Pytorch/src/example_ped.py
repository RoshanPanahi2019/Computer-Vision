import os
import numpy as np
import torch
import torchvision
import utils
import transforms as T
from torchvision.transforms import ToPILImage
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor 
from engine import train_one_epoch, evaluate
from PIL import Image
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class myDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
   
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
 
         # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # image = Image.fromarray(masks[i][ymin:ymax,xmin:xmax],'L') # Visualize the masked region 

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        print(masks[0])
        # m=ToPILImage()(masks)[0]
        # m.show()
        # (pil_to_tensor.squeeze_(0)))
        exit()
        print("xxxxxxxxxxxxxxxxxxxxxxxx")
        exit()
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
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
    #writer.add_graph(net, images)
    #writer.close()
    writer.add_image('four_modular_images', img_grid)    # write to tensorboard

def main():
    my_root="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/example_ped/PennFudanPed"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device is:")
    print(device)
    num_classes = 2 #  dataset has three classes: {background, Module, BathPod}
    dataset = myDataset(my_root, get_transform(train=True))     # use our dataset and defined transformations
    dataset_test = myDataset(my_root, get_transform(train=False))
    #indices = torch.randperm(len(dataset)).tolist()    # split the dataset in train and test set --> Randomize
    indices = [*range(0,(len(dataset)))]   # split the dataset in train and test set --> Don't Randomize

    my_dataset = torch.utils.data.Subset(dataset, indices[:-50])
    my_dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn) # define training and validation data loaders
    data_loader_test = torch.utils.data.DataLoader(
        my_dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
 
    model = get_model_instance_segmentation(num_classes)     # get the model using our helper function
    model.to(device)     # move model to the right device
    params = [p for p in model.parameters() if p.requires_grad]     # construct an optimizer
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)     # and a learning rate scheduler
    #myVisualize(dataset)
    
    num_epochs = 10   # let's train it for 10 epochs
    my_mAP_Seg=[]
    my_mAP_bbox=[]
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1) # train for one epoch, printing every 10 iterations
        lr_scheduler.step()         # update the learning rate
        img_write=False  # evaluate on the test dataset
        if (epoch==9):img_write=True
        print("evaluating===============================================")
        coco_evaluator=evaluate(model, data_loader_test,img_write, device=device)
        #TODO: The reasson the Loss explodes: Probably because some pixels have two values? Where the object is copied on previous objects. 
        # TODO: 

        # mAP_bbox=coco_evaluator.coco_eval['bbox'].stats[0] # mAP for IoU=0.50:0.95 
        # mAP_seg=coco_evaluator.coco_eval['segm'].stats[0] # mAP for IoU=0.50:0.95 
        # my_mAP_Seg.append(mAP_seg)
        # my_mAP_bbox.append(mAP_bbox)

    # epochs = range(0, num_epochs)  
    # plt.plot(epochs,my_mAP_Seg)
    # plt.show()
    # plt.plot(epochs,my_mAP_bbox)
    # plt.show()
    #TODO: mAP is very low: add more images, Create a test set, do the augmentation. 

        # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.bar(class_names, mAP)
    # plt.xticks(rotation=90)
    # plt.ylim([0, 1])
    # plt.ylabel("mAP")
    # plt.show()
        
    print("That's it!")

if __name__ == "__main__":
    main()