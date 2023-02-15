# https://towardsdatascience.com/mask-rcnn-implementation-on-a-custom-dataset-fd9a878123d4
import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

def single_class_mask():
            
    with open("box_json.json", "r") as read_file:
        data = json.load(read_file)

    all_file_names=list(data.keys())
    
    Files_in_directory = []
    for root, dirs, files in os.walk("images"):
        for filename in files:
            Files_in_directory.append(filename)

    for j in range(len(all_file_names)): 
        image_name=data[all_file_names[j]]['filename']
        if image_name in Files_in_directory: 
            img = np.asarray(PIL.Image.open('images/'+image_name))    
        else:
            continue
        mask = np.zeros((img.shape[0],img.shape[1]))
        for i in range (len(data[all_file_names[j]]['regions'])):     
            if data[all_file_names[j]]['regions'] != {}:
                shape1_x=data[all_file_names[j]]['regions'][i]['shape_attributes']['all_points_x']
                shape1_y=data[all_file_names[j]]['regions'][i]['shape_attributes']['all_points_y']
                #fig = plt.figure()
                #plt.imshow(img.astype(np.uint8)) 
                #plt.scatter(shape1_x,shape1_y,zorder=2,color='red',marker = '.', s= 55)
                ab=np.stack((shape1_x, shape1_y), axis=1)
    
                img2=cv2.drawContours(img, [ab], -1, (i+1,i+1,i+1), -1)
            
                #mask = np.zeros((img.shape[0],img.shape[1]))
                img3=cv2.drawContours(mask, [ab], -1, i+1, -1)
        k=j+1        
        cv2.imwrite('masks/%01.0f' % k +'.png',mask.astype(np.uint8))

def multi_class_mask():
    path="/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/annotation/18/via_project_13Feb2023_12h25m_json.json" #  Walk the directory and include all files. 
    with open(path, "r") as my_json:
        json_per_image=json.load(my_json)
    for key in json_per_image.keys():
        image_name=json_per_image[key]['filename'] 
        img = np.asarray(PIL.Image.open('/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/images/'+image_name))    
        
        polygons = [r['shape_attributes'] for r in json_per_image[key]['regions']] 
        objects = [s['region_attributes']['Component'] for s in json_per_image[key]["regions"]]
        num_ids = [name_dict[a] for a in objects]
        print("objects:",objects)
        print("numids:",num_ids)

        for i in range (len(objects)):
            shape1_x=json_per_image[key]["regions"][i]['shape_attributes']['all_points_x']
            shape1_y=json_per_image[key]["regions"][i]['shape_attributes']['all_points_y']
            #lbl=label[json_per_image[key]["regions"][i]["region_attributes"]["Component"]] # Return the label for each region
            ab=np.stack((shape1_x, shape1_y), axis=1)
            mask = np.zeros((img.shape[0],img.shape[1])) # draw a blank canvas
            img3=cv2.drawContours(mask, [ab], -1, num_ids[i]*50, -1) 
            cv2.imwrite('/media/mst/Backup/dataset/InstanceSegmentation/myDataset_Augmentation/masks/' +image_name[:-5]+"-"+str(i)+'.png',mask.astype(np.uint8))

#==================================
if __name__=="__main__":
    #single_class_mask()
    name_dict={"Module":1,"BathPod":2}
    color={}
    multi_class_mask()
