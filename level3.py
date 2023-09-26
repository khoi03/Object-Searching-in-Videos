from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv
from PIL import Image
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from kornia.feature import LoFTR

class OSV3:
    def __init__(self, model_name, video_path, output_path, input_img, image_type):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.video_path = video_path #"Example video.mp4"
        self.model = self.load_model()

        self.tracked_id = -1
        self.max_matches = 0
        self.flag = 0
        
        self.input_img = input_img
        self.output_path = output_path
        self.image_type = image_type
        
        self.matcher = self.load_feature_matching_model()
        
    def load_model(self):
        model = YOLO(self.model_name) #'yolov8m.pt'
        model.to(self.device)

        return model

    def load_feature_matching_model(self):
      matcher = LoFTR(pretrained=self.image_type).eval().to(self.device)
      
      return matcher

    def track(self, frame):
        return self.model.track(frame, persist=True, verbose=False)

    def predict(self, frame):
        return self.model(frame, verbose=False)

    def feature_matching(self, img0_raw, img1_raw):
      img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
      img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
      batch = {'image0': img0, 'image1': img1}

      with torch.no_grad():
        matches = self.matcher(batch)

      return matches['confidence'].cpu().numpy()


    def plot_boxes(self, results, frame):
        for r in results:
            annotator = Annotator(frame)
            result = r.boxes.cpu()
            objects_id = result.id
            goal_object_idx = list(r.names.keys())[list(r.names.values()).index('person')]
            location_idx = np.where(result.cls == goal_object_idx)

            for i in location_idx[0]:
                if torch.any(objects_id == self.tracked_id):
                    index = torch.where(objects_id == self.tracked_id)[0].item()
                    b = result.xyxy[index]
                    annotator.box_label(b, 'wanted', color=(0, 0, 255), txt_color=(255, 255, 255))
                    self.flag = 1
                else:
                    b = result.xyxy[i]
                    object_id = objects_id[i]
                    x1, x2 = int(b[1]), int(b[3])
                    y1, y2 = int(b[0]), int(b[2])

                    gray_input_object = cv.cvtColor(self.input_img, cv.COLOR_BGR2GRAY)
                    gray_object_extract = cv.cvtColor(frame[x1:x2, y1:y2], cv.COLOR_BGR2GRAY)
                    gray_input_object = cv.resize(gray_input_object,(640, 480))
                    gray_object_extract = cv.resize(gray_object_extract,(640, 480))

                    matches = self.feature_matching(gray_input_object, gray_object_extract)
                    matches_count = len(np.where(matches>0.5)[0])
                    
                    # if matches_count > self.max_matches:
                    #   self.max_matches = matches_count
                    #   print(self.max_matches)
                    if matches_count > 65:
                        self.tracked_id = object_id
                        annotator.box_label(b, 'wanted', color=(0, 0, 255), txt_color=(255, 255, 255))
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
            results = self.track(frame)

            # Visualize the results on the frame
            annotated_frame = self.plot_boxes(results,frame)
            
            output_path = self.output_path + '/Frame {}.jpg'.format(i)
            if self.flag == 1:
                cv.imwrite(output_path, annotated_frame)
                self.flag = 0
                
            process.update(1)
        cap.release()


if __name__ == '__main__':
    input_img = cv.imread('Input/Level 3.jpg')
    lvl3 = OSV3('yolov8l.pt', 'Videos/Video 3.mp4', './OutputFrame/Level3/', input_img, 'outdoor')

    lvl3()