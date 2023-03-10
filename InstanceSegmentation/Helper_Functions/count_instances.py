# Counts the number of instances in each mask folder, per class. 

import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

def multi_class_mask():
    annotations="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/annotation/"
    count_module=0
    count_bathpod=0
    
    for i in range(len(os.listdir(annotations))):
        if i<31:
            path="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/annotation/"+str(i+1)+".json" # TODO - Walk the directory and include all files. 
            with open(path, "r") as my_json:  # read in the json annotation file, by VIA
                json_per_image=json.load(my_json)

            for key in json_per_image.keys(): 
                objects = [s['region_attributes']['Component'] for s in json_per_image[key]["regions"]]
                num_ids = [name_dict[a] for a in objects]
                print("objects:",objects)
                print("numids:",num_ids)
                
                count_module=count_module+num_ids.count(1)
                count_bathpod=count_bathpod+num_ids.count(2)
                print("number of module instances:"+str(count_module))
                print("number of bathpod instances:"+str(count_bathpod))

                
#==================================
if __name__=="__main__":
    name_dict={"Module":1,"Bath_Pod":2}
    color={}
    multi_class_mask()
