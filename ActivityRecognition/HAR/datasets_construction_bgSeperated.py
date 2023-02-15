import os
import sys
import tarfile
import collections
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import json
import torch.utils.data as data
import torch
from torch.nn import functional as F
import numpy as np
import pandas
from PIL import Image
from torchvision import transforms
import random 

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.225, 0.225, 0.225])

def read_gt(gt_file, nb_class, exclude_bg):
    
    print('-----------------------reading gt---------------------------')
  
    class_to_id = {}
    data = {}
    l = pandas.read_excel(gt_file)

    count_class = 0
    #print(l.columns)
    for i, c in enumerate(l.columns): 
        if count_class >= nb_class: break
        if exclude_bg:
            if i == 0: continue
        id_class, class_name = c.split('(')
        class_name = class_name[:-1]

        key = int(id_class)

        #if key == 0: continue
        if exclude_bg: 
            key = key - 1
        
        class_to_id[key] = class_name

        count_class += 1
        #print('key {}'.format(key))
        for bbox in list(l[c]):   
 
            if type(bbox) == float: continue   
            window, pid = bbox.replace(' ', '').split(',')

            start, end = window.split('-')

           
            start, end, pid = int(start), int(end), int(pid)
            if pid not in data: data[pid] = []
            data[pid].append((start, end, key))
      
    #exit()
    return data, class_to_id

def read_tracking(tr_file):
    with open(tr_file, 'r') as f:
        tracks = f.readlines()

    track_dict = {}    
    for tr in tracks:
        fr, bbox_id, x, y, w, h, _, _, _, _ = tr.split()
        fr, bbox_id, x, y, w, h = int(fr), int(bbox_id), int(x), int(y), int(w), int(h)
        if bbox_id not in track_dict: track_dict[bbox_id] = {}
            
        track_dict[bbox_id][fr] = [x, y, w, h]
    return track_dict

def fuse(gt, track, video_path, exclude_bg):
    data = []
    pid_data = {}
    for pid, windows in gt.items():
        if not pid in track: continue
        frame_data = {}
        for frame, bbox in track[pid].items():
            x, y, w, h = bbox
            isFound = False
            img = os.path.join(video_path, "frame" + str(frame) + '.jpg')
            for win in windows:
                start, end, label_id = win
                if start <= frame <= end:
                    isFound = True         
                    data.append((frame, img, label_id, x, y, w, h, pid))
                 
                    if pid not in pid_data: pid_data[pid] = {}
                    pid_data[pid][frame] = (frame, img, label_id, x, y, w, h, pid)
                    break
            if exclude_bg: continue
            if not isFound:
                
                data.append((frame, img, 0, x, y, w, h, pid))
                if pid not in pid_data: pid_data[pid] = {}
                pid_data[pid][frame] = (frame, img, 0, x, y, w, h, pid)  

    return data, pid_data


    
class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return "" 

class ConstructionDataset(VisionDataset):
    def __init__(self,
                 root,
                 image_set='train',
                 patch_size=128,
                 ratio=2.0,
                 scale=5.0,
                 mask=False,
                 small=True,
                 num_classes=5,
                 exclude_bg = False):
        super().__init__(root)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        self.exclude_bg = exclude_bg
        self.videos = []
        
        for x in os.listdir(root):
            #print(root)
            #[,,nailing]
            if image_set == 'train':           
                if x in ['video0','video1','video2','video3','video4','video5','video6','video7','video8','video10','video12','video13' ]:#'video9' 'video8'   #'video1','video2','video4,,'video38','video13',
                    self.videos.append(os.path.join(root, x))
                    
            elif image_set == 'test':
                #[SitSnaling, nailing, cutting, watering]
                
                if x in ['video101','video15','video3','video55']:#,'video55']:#,'video15','video0']:#,"Nailing_test_1"]:#'Framing4','video7']:
                    self.videos.append(os.path.join(root, x)) 
                    # print('these are the videos...............')
                    #print(self.videos)
                    
            elif image_set == 'val':
                if x in ['video43']:#'Framing4']:
                    self.videos.append(os.path.join(root, x))         
                    
        self.video_data = {}
        self.data = []
        self.video_data = {}
        #print(self.videos)
        
        for i, video in enumerate(self.videos):     
           

            video_name = video.replace('\\', '/').split('/')[-1]
            video_path = os.path.join(video, video_name).replace('\\', '/')
            gt_data, class_to_id = read_gt(os.path.join(video, video_name + '.xlsx'), num_classes, exclude_bg)
            # print('-----------------------reading gt---------------------------')
           # print(gt_data)
           
            track_data = read_tracking(os.path.join(video, video_name + '.txt'))
            data, pid_data = fuse(gt_data, track_data,os.path.join(video, video_name), exclude_bg)
            self.data.extend(data)
            if video_path not in self.video_data: self.video_data[video_path] = {}
            self.video_data[video_path] = {**self.video_data[video_path], **pid_data} 
        # exit()
        activity_count = {}
        for _, _, label, _, _, _, _, _ in self.data:

            #exit()
            if label not in activity_count:
                activity_count[label] = 0
            activity_count[label] += 1
        print(activity_count.keys())
        print('-----------------------=======================')
        print(activity_count)
        
        #class_to_id[0] = 'background'
        #self.class_to_id = class_to_id
        self.ratio = ratio
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.mask = mask
        self.scale = scale 
        print('num {} examples'.format(image_set), len(self.data))
      
    def set_data(self, window):
        self.data = window
        
    def __getitem__(self, index):
        frame, image_path, label, x, y, w, h, pid = self.data[index]
        #_, image_path, label = self.data[index]
        img = Image.open(image_path).convert('RGB')
        
        img = np.array(img)
        H, W = img.shape[:2]
        if x == 0 and y == 0:
            x, y, w, h = int(H/4),int(W/4), int(H/2), int(W/2)

        b_patch = self.transform(img[y:y+h, x:x+w])

        if self.mask:
            mean_rgb = img.reshape((-1, 3)).mean(0)
            img[y:y+h, x:x+w] = mean_rgb

        c_patch = self.transform(img)

        return frame, b_patch, c_patch, label, pid

    def __len__(self):
        return len(self.data)

#if __name__ == "__main__":
 
    #dataset = ConstructionDataset("D:/dataset/Test/Video0") 
                                    