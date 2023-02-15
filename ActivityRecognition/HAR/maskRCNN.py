import os, time
from tqdm import tqdm
import torch, torchvision
import numpy as np
import PIL
import cv2
os.environ['TORCH_HOME'] = 'models/'

CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 20 - 1, 2 ** 10 - 1, 2 ** 15 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"]
    labels = predictions['labels']
    boxes = predictions['boxes']
    labels = [CATEGORIES[i] if i < 80 else CATEGORIES[0] for i in labels]

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions['labels']
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 3
        )
    
    return image

def load_data(video_path):
    frames = [os.path.join(video_path, x) for x in os.listdir(video_path) if x.endswith('.jpg')]
    return frames

def load_video(video_path):
    return cv2.VideoCapture(video_path)
    
    
def tracking(model, dataset, output_file, save = False, bs = 1,nbframe = 1000000):
    

    n = 0
    k=0
    
    #output_f = output_file + 'output.csv'

    output_f= output_file +'.csv'
    with open(output_f, 'w') as f:
        f.write('')
    #video = cv2.VideoCapture(DATADIR+file_name);
    ret, frame = dataset.read()
    H,W,_ = np.array(frame).shape
    #frame = cv2.resize(frame, (H//2,W//2), interpolation = cv2.INTER_AREA)
    
    if ret== False: return
    
    # we do not want to write the videos
    if save:        
        H,W,_ = np.array(frame).shape
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fps = video.get(cv2.CAP_PROP_FPS)
   
        video_writer = cv2.VideoWriter(output_file+'.avi', fourcc,30, (W, H))
        
    time_list = []
    start = time.time()  
    count = 0
    with torch.no_grad():
        while n < nbframe:
            image_tensor = []
            if save: image = []
            for b in range(bs):              
                im = np.array(frame)
                if save: image.append(im)
                image_tensor.append(torchvision.transforms.functional.to_tensor(im).cuda())
                ret, frame= dataset.read()
                if ret== False:
                    break
                n+=1
 
            print(image_tensor[0].size())
            output = model(image_tensor)
            print('Complete tracking')
            for i, o in enumerate(tqdm(output)):
                
                if save: new_o = {'boxes':[], 'labels':[], 'masks':[], 'scores':[]}
                for label, score, bb, m in zip(o['labels'], o['scores'], o['boxes'], o['masks']):
              
                    if label == 1 and score > 0.3:
                        x_top, y_top, x_bot, y_bot = bb.cpu().tolist()
                        w = x_bot - x_top
                        h = y_bot - y_top
                        with open(output_f, 'a') as f:
                            f.write('{},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1\n'.format(count, x_top, y_top, w, h, 100*score.cpu()))
                        if save:
                            new_o['labels'].append(label.cpu())
                            new_o['scores'].append(score.cpu())
                            new_o['boxes'].append(bb.cpu())
                            new_o['masks'].append(m.cpu())
                if len(new_o['labels'])!=0:    
                    if save:                       
                        new_o['labels'] = torch.stack(new_o['labels'],0 )
                        im = overlay_boxes(image[i], new_o)
                        im = overlay_class_names(im, new_o)
                        im=image[i]
                        
                        
                        
                        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
                        #cv2.imshow(im)
                        #print(output_file, os.path.join(output_file  + "/", '{}.jpg'.format(i) ), os.path.exists(os.path.join(output_file  + "/", '{}.jpg'.format(i) )))  
                       # if not os.path.exists(output_file  + "/"):
                            #os.mkdir(output_file + "/")
                        #cv2.imwrite(os.path.join(output_file  + "/", '{}.jpg'.format(n+i-2) ),im)
    
                        video_writer.write(im)
                else:
                    k=k+1
                    print(k)
                count += 1
           
                            
            del output, image_tensor
    dataset.release()        
    if save: video_writer.release()
#D:\Videos\Ceilling\video_1\New_Nailing\Nailing_test_Mute
if __name__=="__main__":
    # DATADIR = 'D:/Videos/Ceilling/video_1/New_Nailing/Nailing_train_3/'
    # file_name='Nailing_train_3.MP4'
    
    
    DATADIR="D:/Videos/Ceilling/video_1/PolishingRoof/Detection/validate/video1/"
   
    #file_name="test_1.avi"
    Output_Dir="D:/Videos/Ceilling/video_1/PolishingRoof/Detection/validate/video1/detection/"

    j=-1
    for file_name in os.listdir(DATADIR):
        if file_name.endswith('.MP4'):   
            #if file_name=="Station3.mp4":
            j+=1
            dataset = load_video(DATADIR+file_name)
            ret, frame = dataset.read()
            #print(ret)
            # while ret==True:
                # cv2.imshow('frame',frame)
                # cv2.waitKey(1)
        #Load mask RCNN pretrained model
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda()
            model.eval()
            output=Output_Dir+file_name+"New"
            try:
                tracking(model, dataset, output , save = True, bs = 1)
            except:
                continue

                #tracking(model, dataset, '/scratch/aziere/SportDataset/test/', save = False, bs = 5)




