from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv 
import numpy as np
import torch
import os
from tqdm import tqdm

class OSV1:
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
        
        self.flag = 0
        
    def load_model(self):
        model = YOLO(self.model_name) #'yolov8m.pt'
        model.to(self.device)
        
        return model
    
    def predict(self, frame):
        return self.model(frame, verbose = False)
    
    def plot_boxes(self, results, frame):
        for r in results:
            annotator = Annotator(frame)
            
            result = r.boxes.cpu()
            
            goal_object_idx = list(r.names.keys())[list(r.names.values()).index('truck')]
            location_idx = np.where(result.cls == goal_object_idx)
                
            for i in location_idx[0]:
                b = result.xyxy[i]  
                annotator.box_label(b, 'truck', color=(0, 0, 255), txt_color=(255, 255, 255))
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
            
            output_path = self.output_path + '/Frame {}.jpg'.format(i)
            if self.flag == 1:
                cv.imwrite(output_path, annotated_frame)
                self.flag = 0
                
            process.update(1)
            
        cap.release()

if __name__ == '__main__':
    lvl1 = OSV1('yolov8n.pt', 'Videos/Video 1.mp4', './OutputFrame/Level1/')
    lvl1()