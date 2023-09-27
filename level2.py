from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv 
from PIL import Image
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class OSV2:
    def __init__(self, model_name, video_path, output_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.video_path = video_path #"Example video.mp4"
        self.model = self.load_model()
        
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.location = (0,50)
        self.distance = (0,10)
        self.fontScale = 1
        self.fontColor = (0,0,255)
        self.lineType = 3
        
        self.output_path = output_path
        
        self.thresholds = {
            "blue":(0,50),
            "green":(0,50),
            "red":(120,255)
        }
        self.flag = 0
        
    def load_model(self):
        model = YOLO(self.model_name) #'yolov8m.pt'
        model.to(self.device)
        
        return model
    
    def predict(self, frame):
        return self.model(frame, verbose = False)
    
    def check_range(self, img_value, threshold_min, threshold_max):
        return ((img_value >= threshold_min) & (img_value <= threshold_max)).astype(int)
    
    def color_mask(self, threshold, img):
        b_min, b_max = threshold["blue"]
        g_min, g_max = threshold["green"]
        r_min, r_max = threshold["red"]
    
        b_mask = self.check_range(img[:,:,0],b_min, b_max)
        g_mask = self.check_range(img[:,:,1],g_min, g_max)
        r_mask = self.check_range(img[:,:,2],r_min, r_max)
        
        mask = b_mask * g_mask * r_mask 
        h,w = mask.shape

        return mask.reshape(1,h,w)


    def plot_boxes(self, results, frame):
        for r in results:
            annotator = Annotator(frame)
            result = r.boxes.cpu()
            masks = r.masks
            goal_object_idx = list(r.names.keys())[list(r.names.values()).index('truck')]
            location_idx = np.where(result.cls == goal_object_idx)
            
            for i in location_idx[0]:
                b = result.xyxy[i]  
                object_mask = masks[i].data.cpu().numpy()
                object_mask = object_mask.astype('uint8')
                object_mask_resize = cv.resize(object_mask[0],(frame.shape[1],frame.shape[0]))
                x1, x2 = int(b[1]), int(b[3])
                y1, y2 = int(b[0]), int(b[2])
                
                object_mask_resize = object_mask_resize[x1:x2, y1:y2]
                object_extract = cv.bitwise_and(frame[x1:x2, y1:y2], frame[x1:x2, y1:y2], mask=object_mask_resize)
                related_color_mask = self.color_mask(self.thresholds, object_extract)
                
                related_color_number = np.count_nonzero(related_color_mask)
                object_pixel_number = np.count_nonzero(object_mask_resize)
                
                threshold_number = related_color_number / object_pixel_number
                if (threshold_number > 0.2):
                    annotator.box_label(b, 'red truck', color=(0, 0, 255), txt_color=(255, 255, 255))
                    self.flag = 1
            
        annotated_frame = annotator.result()  
                
        return annotated_frame
    
    def __call__(self):
        cap = cv.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        process = tqdm(total=total_frames)
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        i = 0
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                break
            i += 1
            # Run YOLOv8 inference on the frame
            results = self.predict(frame)

            # Visualize the results on the frame
            annotated_frame = self.plot_boxes(results,frame)
            # cv.imshow('YOLO V8 Detection', annotated_frame)
            # if cv.waitKey(1) & 0xFF == ord(' '):
            #     break
            output_path = self.output_path + '/Frame {}.jpg'.format(i)
            if self.flag == 1:
                cv.imwrite(output_path, annotated_frame)
                self.flag = 0
            process.update(1)
            
        cap.release()
    
    
if __name__ == '__main__':
    lvl2 = OSV2('yolov8l-seg.pt', './Videos/Video 1.mp4', 'OutputFrame/Level2')
    lvl2()