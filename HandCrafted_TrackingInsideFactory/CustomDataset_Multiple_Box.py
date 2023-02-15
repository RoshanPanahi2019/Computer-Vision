
import os
import torch
from skimage import io
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import pandas as pd
import time

class BoxTrackingDataset(Dataset):
    def __init__(self, csv_file_Support,csv_file_RoI, video, support, batch_size,step_frame, transform=None, test = False):
        self.video_path = video
        self.video = cv2.VideoCapture(video)
        self.image_dir = {}
        with open(csv_file_Support, 'r') as f:
            annot = f.readlines()    
            for i, line in enumerate(annot):
                if i > 4: break
                line = line.split(',')
                image = line[0]
                label = int(line[1].replace('\n', ''))
                self.image_dir[i] = os.path.join(support, image)
        camera_layout=pd.read_csv(csv_file_RoI)  

        self.transform = transform
        self.current_frame = 0
        self.total_frame = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.batch_size = batch_size
        self.step_frame=step_frame
        self.box_1=[300, 800, 800, 450 ]
        self.box_2=[300, 200, 900, 350 ]
        self.test = test

    def get_support_images(self, raw = False):
        print(self.image_dir)
        print('--------------------')

        image0 = io.imread(self.image_dir[0])
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image1 = io.imread(self.image_dir[1])
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = io.imread(self.image_dir[2])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = io.imread(self.image_dir[3])
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        y_label_1 = torch.tensor([0,1])
        y_label_2 = torch.tensor([0,1])

        if image0.shape[2] == 4: image0 = image0[:, :, :3]
        if image1.shape[2] == 4: image1 = image1[:, :, :3]
        if image2.shape[2] == 4: image2 = image2[:, :, :3]
        if image3.shape[2] == 4: image3 = image3[:, :, :3]       
        if self.transform and not raw:           
            image0 = self.transform(image0)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)            
            
        else:
            image0 = torch.Tensor(image0)   
            image1 = torch.Tensor(image1)
            image2 = torch.Tensor(image2)   
            image3 = torch.Tensor(image3)           
                                  
        image_1 = torch.stack([image0,image1], dim = 0)
        image_2 = torch.stack([image2,image3], dim = 0)

        return (image_1, y_label_1,image_2,y_label_2)    

    def predict_video(self, device, model, classifier, transform_test, output = 'output_video.avi'):
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        video_out = cv2.VideoWriter(output, fourcc, fps, (frame_width,frame_height))
        
        self.video = cv2.VideoCapture(self.video_path)
        success = True
        
        count = 0
        position = (10, frame_height - 10)
        scale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        with torch.no_grad():
            support, target_support = self.get_support_images()
            support_features = model(support.to(device))
            support_features = F.normalize(support_features, p=2, dim = -1)
            while success:
                success, image = self.video.read()
                if success:
                    x = image[self.box_1[0]: self.box_1[0] + self.box_1[2], self.box_1[1]: self.box_1[1] + self.box_1[3], :]
                    x = transform_test(x).unsqueeze(0)
                    x = model(x.to(device))
                    x = F.normalize(x, p=2, dim = -1)
                    x = torch.mm(x, support_features.t()) # If use support images
                    print(x)
                    x = torch.argmax(x, dim = -1).squeeze(0).cpu().numpy()
                    if x == 0:
                        message = "NO box_1 FOUND IN THIS STATION"
                    else:
                        message = "box_1 FOUND IN THIS STATION"

                    cv2.putText(image, message, position, font, scale , (255, 255, 255), 2)
                    video_out.write(image)
                    count += 1
        print("Prediction finished")
        self.video.release()
        video_out.release()
    
    def __del__(self):
        self.video.release()
               
    def __len__(self):
        return int((self.total_frame // self.batch_size)//self.step_frame) + 1
    
    #@profile
    def __call__(self, index): 
        clip_1 = []
        clip_2 = []
        for i in range(self.batch_size):
            for j in range(self.step_frame):
                success, image = self.video.read()
                if not success:
                    self.video.release()
                    if self.test:
                        return torch.stack(clip, dim = 0)

                    break
                if not success: 
                    print("Problem reloading video")
                    exit()

            if not success: break
            image_1 = image[self.box_1[0]: self.box_1[0] + self.box_1[2], self.box_1[1]: self.box_1[1] + self.box_1[3], :]
            image_2 =image[self.box_2[0]: self.box_2[0] + self.box_2[2], self.box_2[1]: self.box_2[1] + self.box_2[3], :]
            clip_1.append(torch.Tensor(image_1))
            clip_2.append(torch.Tensor(image_2))

        if len(clip_1) == 0: return [], []
        torch.stack(clip_1, dim = 0)
        torch.stack(clip_2, dim = 0)
        return clip_1,clip_2
    
    def __getitem__(self, index): 
        clip = []
        for i in range(self.batch_size):
            success, image = self.video.read()
            if not success:
                self.video.release()
                if self.test:
                    return torch.stack(clip, dim = 0)
                self.video = cv2.VideoCapture(self.video_path)
                success, image = self.video.read()
                self.current_frame = 0

            if not success: 
                print("Problem reloading video")
                exit()

            image = image[self.box_1[0]: self.box_1[0] + self.box_1[2], self.box_1[1]: self.box_1[1] + self.box_1[3], :]
            self.current_frame += 1
            if self.transform:
                image = self.transform(image)
            clip.append(image)
        return torch.stack(clip, dim = 0)

