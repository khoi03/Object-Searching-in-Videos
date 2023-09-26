# Object-Searching-in-Videos

## Introduction
In this task, I will focus on three different levels of searching for objects within videos:
- **Level 1:** Find similar objects with no properties: `A truck`.
- **Level 2:** Find object with color property: `The red truck`.
- **Level 3:** `Find this person`

Eventually, I locate all frames and draw bounding boxes around the finding object X in the videos, and then export these frames as JPG files. The structure of the output folders should be as follows:
- Video 1
  - Object X
    - Frame 15.jpg
    - Frame 32.jpg
    - Frame 120.jpg
- Video 2
  - Object X
- Video 3
  - Object X
    - Frame 215.jpg


## Table of contents:

1. [Approach](https://github.com/khoi03/Object-Searching-in-Videos#1-approach)

    1. [Level 1](https://github.com/khoi03/Object-Searching-in-Videos#i-level-1)
    
    2. [Level 2](https://github.com/khoi03/Object-Searching-in-Videos#ii-level-2)
    
    3. [Level 3](https://github.com/khoi03/Object-Searching-in-Videos#iii-level-3)
     
2. [Results](https://github.com/khoi03/Object-Searching-in-Videos#2-results)

## 1. Approach
### i. Level 1: 
At this level, I employ `YOLOv8` model to detect all objects in the video, and subsequently I extract and draw bounding boxes exclusively around objects classified as `truck`.

### ii. Level 2:
Moving to the next stage, I commence by replicating the procedures of Level 1. utilizing `YOLOv8` model to extract `truck` objects. Furthermore, in this task I employ the large segmentation YOLOv8 model(yolov8l-seg.pt) for all three levels. This choice is made not only to enhance the prediction accuracy due to its larger size but also it has the capability to generate masks for the detected objects as follows:

Then apply color detection algorithm on the detected object.

### iii. Level 3:
At this final stage, I incorporate the use of the `YOLOv8` model and [LoFTR](https://github.com/zju3dv/LoFTR) model.

## 2. Results
