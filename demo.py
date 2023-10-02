import os
from level1 import OSV1
from level2 import OSV2
from level3 import OSV3
import cv2 as cv

model_name = 'yolov8l-seg.pt'
video_folder = './Videos/'
output_folder = './OutputFrame/'

for file_name in os.listdir(video_folder):
    print(file_name)
    head_name = os.path.splitext(file_name)[0]
    
    input_path = os.path.join(video_folder, file_name)
    output_path = os.path.join(output_folder, head_name)
    
    #level 3
    if file_name == 'Video 3.mp4':
        output_path3 = output_path + '/Object Target Person'
        input_img = cv.imread('./Input/Level 3.jpg')
        lvl3 = OSV3(model_name, input_path, output_path3, input_img, 'outdoor')

        lvl3()
    else:
        #level 1
        output_path1 = output_path + '/Object Truck'
        lvl1 = OSV1(model_name, input_path, output_path1)
        lvl1()
        
        #level 2
        output_path2 = output_path + '/Object Red Truck'
        lvl2 = OSV2(model_name, input_path, output_path2)
        lvl2()
    
    
