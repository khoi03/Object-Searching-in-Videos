# Object-Searching-in-Videos

## Introduction
In this task, I will focus on three different levels of searching for objects within videos:
- **Level 1:** Find similar objects with no properties: `A truck`.
  
  ![Level 1](Input/Level%201.jpg)

- **Level 2:** Find object with color property: `The red truck`.
  
  ![Level 2](Input/Level%202.jpg)

- **Level 3:** `Find this person`
  
  ![Level 3](Input/Level%203.jpg)

Eventually, I locate all frames and draw bounding boxes around the finding object X in the videos, and then export these frames as JPG files. 
<details>
<summary>The structure of the output folders is as follows:</summary>
  
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
</details>

## Table of contents:

1. [How to run this repository](https://github.com/khoi03/Object-Searching-in-Videos#1-how-to-run-this-repository)
   
2. [Approach](https://github.com/khoi03/Object-Searching-in-Videos#2-approach)

    1. [Level 1](https://github.com/khoi03/Object-Searching-in-Videos#i-level-1)
    
    2. [Level 2](https://github.com/khoi03/Object-Searching-in-Videos#ii-level-2)
    
    3. [Level 3](https://github.com/khoi03/Object-Searching-in-Videos#iii-level-3)

3. [Results](https://github.com/khoi03/Object-Searching-in-Videos#3-results)
   
    1. [Level 1](https://github.com/khoi03/Object-Searching-in-Videos#i-level-1-1)
    
    2. [Level 2](https://github.com/khoi03/Object-Searching-in-Videos#ii-level-2-1)
    
    3. [Level 3](https://github.com/khoi03/Object-Searching-in-Videos#iii-level-3-1)

## 1. How to run this repository
I recommend creating an anaconda environment:
```
conda create --name [environment-name] python=3.9
```

Then, install Python requirements:
```
pip install -r requirements.txt
```
Finally, to reproduce the results, you first have to download the provided example videos [here](https://uithcm-my.sharepoint.com/:f:/g/personal/20521482_ms_uit_edu_vn/Eo_svNSYVghLkWb_zqoVVE0BEpb-Bzmp5NDC3_3Wh1eB-g?e=2ZoVOa). Then from the `[environment-name]` project root, run:
```
python demo.py
```

## 2. Approach
### i. Level 1
At this level, I employ `YOLOv8` model to detect all objects in the video, and subsequently I extract and draw bounding boxes exclusively around objects classified as `truck`.

### ii. Level 2
Moving to the next stage, I commence by replicating the procedures of Level 1. utilizing `YOLOv8` model to extract `truck` objects. Furthermore, in this task I employ the large segmentation YOLOv8 model(yolov8l-seg.pt) for all three levels. This choice is made not only to enhance the prediction accuracy due to its larger size but also it has the capability to generate masks for the detected objects, for example:

- Identified Object:
  
  ![object](Media/object.jpg)

- Object's mask:
  
  ![mask](Media/mask.jpg)

In the event that the background contains elements with a similar color to the object, I further enhance accuracy by extracting the detected object based on its mask and applying a color detection algorithm as follows:

- Extracted object:
  
  ![extract](Media/extract.jpg)

To determine whether the pixel values of the object fall within the red color range, I check if the values for the blue and green channels are in the range (0, 50) and for the red channel are in the range (120, 255). Subsequently, I obtain the following red mask:

- Red mask:
  
  ![cmask](Media/cmask.jpg)

Eventually, I can determine whether the detected truck is red by calculating the ratio of red pixels to the total object's pixels and setting a specific threshold for it.

### iii. Level 3
At this final stage, I incorporate the use of the `YOLOv8` model and [Detector-Free Local Feature Matching with Transformers](https://github.com/zju3dv/LoFTR) model (LoFTR for short), you can find their paper [here](https://arxiv.org/pdf/2104.00680.pdf). 
- The first task follows similar procedures to those of Level 1, but it focuses on `human` class.
- Next step is to identify the similarities between the target person (input for this task) and the detected person. LoFTR identifies and extracts keypoints from the given image and the detected human. It then establishes mappings between pairs of keypoints and provides confidence scores for these pairs, you will have a deeper understanding through the following example:

![LoFTR_example](Media/LoFTR_example.jpg)

- Subsequently, I check if the number of confidence scores greater than 0.5 satisfies a particular threshold (I use a threshold of 65 in my code). Eventually, I employ YOLOv8 model to track ID of the detected human. If the model loses track of the person, the process will start over.
  
## 3. Results
In this section, I will provide an overview of the results from the provided examples, which you can access and download from [here](https://uithcm-my.sharepoint.com/:f:/g/personal/20521482_ms_uit_edu_vn/Eo_svNSYVghLkWb_zqoVVE0BEpb-Bzmp5NDC3_3Wh1eB-g?e=2ZoVOa). Furthermore, please access the result frames for each video level via the following [link](https://uithcm-my.sharepoint.com/:f:/g/personal/20521482_ms_uit_edu_vn/EtuOSkYURstCo4GwnQ2j7TABLZQ2C7PyziCb9Uw2Uqx2HQ?e=3K6lMe).
### i. Level 1
- Video 1:
  - Frame 103: 
    ![Frame 103](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/1fc96bc7-5b0b-46ae-87cd-ab170ca52ccb)

- Video 2:
  - Frame 42:
    ![Frame 42](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/e29add09-4669-4439-b38d-73fc9ee659f0)

### ii. Level 2
- Video 1:
  - Frame 266: 
    ![Frame 266](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/e9c2be78-8fe2-41f4-889b-bf6ec5b90599)

- Video 2:
  - Frame 205:
    ![Frame 205](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/36de9ccb-8ed6-43f3-9f43-fdf2f0ee078d)

### iii. Level 3
At the final level, you may want to see the full video result via [this link](https://uithcm-my.sharepoint.com/:f:/g/personal/20521482_ms_uit_edu_vn/EhD8QhMR-6xDmuSqwLC-8dQBrFGJfjPrY2i63Kl1y8vuAA?e=CSxZuW).
- Video 3:
  - Frame 1115:
    ![Frame 1115](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/59046adc-88da-4ec4-83d2-75d76dd85687)
  - Frame 1898:
    ![Frame 1898](https://github.com/khoi03/Object-Searching-in-Videos/assets/80579165/c4914846-473b-43a2-84ad-1507d1485729)
