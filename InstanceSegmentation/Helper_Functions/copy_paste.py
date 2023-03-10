# Copy and paste masks one at a time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os 
import shutil
# example for augmentation: image 1 has 9 instances. For each instance 25 augmentation folders are generated. In each folder, one of the instances are moved. 
                
 
def augment(mask_root,image_root,path_dataset):
    img_list=os.listdir(image_root)
    for img_name in os.listdir(mask_root):   
        img=Image.open(image_root+"/"+img_name+".jpeg")

        instance_id=0
        for file in os.listdir(mask_root+"/"+img_name):
            mask=Image.open(mask_root+"/"+img_name+"/"+file)
            mask=np.array(mask)
            pos=np.where(mask)

            if (mask.shape[1]-max(pos[1])<min(pos[1])): # We leave one step as padding. 
                step_x=-int(min(pos[1])/6) # slide in left direction 
            else:
                step_x=int((mask.shape[1]-max(pos[1]))/6) # slide in right direction

            if (mask.shape[0]-max(pos[0])<min(pos[0])):
                step_y=-int(min(pos[0])/6) # slide up
            else:
                step_y=int((mask.shape[0]-max(pos[0]))/6) # slide  down

            id=0    
            for i in range(1 , 6):
                for j in range (1,6):
                    canvas = np.zeros((mask.shape[0],mask.shape[1])) # draw a blank canvas
                    canvas[pos[0]+step_y*j,pos[1]+step_x*i]=np.unique(mask)[1] # 0 is the background 
                    canvas=canvas.astype(np.uint8) #convert to PNG compatiable format
                    canvas = Image.fromarray(canvas) #numpy to PIL            
                    img2=np.array(img)
                    img2[pos[0]+step_y*j,pos[1]+step_x*i]=np.array(img)[pos[0],pos[1],:]
                    img2=Image.fromarray(img2)
                    new_mask_folder=path_dataset+"/"+img_name+"-"+str(instance_id)+"-"+str(id)
                    os.mkdir(new_mask_folder)
                    for my_file in os.listdir(mask_root+"/"+img_name): # Copy annotations for the image to the new folder
                        shutil.copy(mask_root+"/"+img_name+"/"+my_file,new_mask_folder)     
                    canvas.save(new_mask_folder+"/"+img_name+"-"+str(id)+"-augmented"+".png") # Add augnented mask to new folder
                    img2.save(path_dataset+"/"+img_name+"-"+str(instance_id)+"-"+str(id)+".jpeg")           
                    id=id+1  
            instance_id=instance_id+1

        # Image coordinate system:
        #  (y,x) x----> 
        #       |y
        #       |
        #       V

        # Directory Tree: 
        # One Mask folder per image. Constains one mask.png per instance in the RGB image. 
        # palce all masks folders next to RGB pairs with the same name as the folder. 
#===========================

if __name__=="__main__":
    path_mask_root_all=[]
    mask_root="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/masks"
    image_root="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/images"
    path_dataset='/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/all_in_onePLace_set_1' 
    num_augment_per_image=25 # number of augmented images created per image using sliding method

    augment(mask_root,image_root,path_dataset)

