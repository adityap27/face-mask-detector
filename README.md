




# face-mask-detector
𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐝𝐞𝐞𝐩𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐀𝐥𝐞𝐫𝐭 𝐬𝐲𝐬𝐭𝐞𝐦 💻🔔


## System Overview

It detects human faces with 𝐦𝐚𝐬𝐤 𝐨𝐫 𝐧𝐨-𝐦𝐚𝐬𝐤 even in crowd in real time with live count status and notifies user (officer) if danger.

<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-airport.gif?raw=true">
</p>

**System Modules:**
  
1. **Deep Learning Model :** I trained a YOLOv2,v3 and v4 on my own dataset and for YOLOv4 achieved **93.95% mAP on Test Set** whereas YOLOv3 achieved **90% mAP on Test Set** even though my test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.  
  
2. **Alert System:** It monitors the mask, no-mask counts and has 3 status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.
	3. **Danger :** ( + SMS Alert ) When _some ratio_ of people are without mask.


## Table of Contents
- [Quick-Start : Just Run Inference](#Quick-Start)
1. [Face-Mask Dataset](#Face-Mask-Dataset)
	1. [Image Sources](#1-Image-Sources)
	2. [Image Annotation](#2-Image-Annotation) 
	3. [Dataset Description](#3-Dataset-Description)
2. [Deep Learning Models](#Deep-Learning-Models)
	1. [Training](#1-Training)
	2. [Model Performance](#2-Model-Performance)
	3. [Inference](#3-Inference)
		1. [Detection on Image](#31-Detection-on-Image)
		2. [Detection on Video](#32-Detection-on-Video)
		3. [Detection on WebCam](#33-Detection-on-WebCam)
3. [Alert System](#Alert-System)
4. [Suggestions to improve Performance](#Suggestions-to-improve-Performance)
5. [References](#References)

## Quick-Start
**Step 1:**
```
git clone https://github.com/adityap27/face-mask-detector.git
```
Then, Download weights. https://bit.ly/yolov4_mask_weights and put in **yolov4-mask-detector** folder

**Step 2: Install requirements.**
```
pip install opencv-python
pip install imutils
```
**Step 3: Run yolov4 on webcam**
```
python mask-detector-video.py -y yolov4-mask-detector -u 1
```
Optional: add ```-e 1``` for Email notifications.
## Face-Mask Dataset

### 1. Image Sources
- Images were collected from [Google Images](https://www.google.com/imghp?hl=en), [Bing Images](https://www.bing.com/images/trending?form=Z9LH) and some [Kaggle Datasets](https://www.kaggle.com/vtech6/medical-masks-dataset).
- Chrome Extension used to download images: [link](https://download-all-images.mobilefirst.me/)

### 2. Image Annotation
- Images were annoted using [Labelimg Tool](https://github.com/tzutalin/labelImg).

### 3. Dataset Description
- Dataset is split into 3 sets:

|_Set_|Number of images|Objects with mask|Objects without mask|
|:--:|:--:|:--:|:--:|
|**Training Set**| 700 | 3047 | 868 |
|**Validation Set**| 100 | 278 | 49 |
|**Test Set**| 120 | 503 | 156 |
|**Total**|920|3828|1073|

- **Download the Dataset here**:
	+ [Github Link](https://github.com/adityap27/face-mask-detector/tree/master/dataset) or
	+ [Kaggle Link](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format)

## Deep Learning Models

### 1. Training
- Install [Darknet](https://github.com/AlexeyAB/darknet) for Mac or Windows first.
- I have trained Yolov2,Yolov3 and YOLOv4.
- Use following (linux) cmd to train:


```console
./darknet detector train obj.data yolo3.cfg darknet53.conv.74
```
- for windows use **darknet.exe** instead of ./darknet

**YOLOv2 Training details**
- Data File = [obj.data](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov2-mask-detector/obj.data)
- Cfg file  = [yolov2.cfg](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov2-mask-detector/yolov2.cfg)
- Pretrained Weights for initialization= [yolov2.conv.23](https://pjreddie.com/media/files/darknet19_448.conv.23)
- Main Configs from yolov2.cfg:
	- learning_rate=0.001
	- batch=64
	- subdivisions=16
	- steps=1000,4700,5400
	- max_batches = 6000
	- i.e approx epochs = (6000*64)/700 = 548
- **YOLOv2 Training results: _0.674141 avg loss_**
-  **Weights** of YOLOv2 trained on Face-mask Dataset: [yolov2_face_mask.weights](https://bit.ly/yolov2_mask_weights)

**YOLOv3 Training details**

- Data File = [obj.data](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov3-mask-detector/obj.data)
- Cfg file  = [yolov3.cfg](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov3-mask-detector/yolov3.cfg)
- Pretrained Weights for initialization= [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
- Main Configs from yolov3.cfg:
	- learning_rate=0.001
	- batch=64
	- subdivisions=32
	- steps=4800,5400
	- max_batches = 6000
	- i.e approx epochs = (6000*64)/700 = 548
- **YOLOv3 Training results: _0.355751 avg loss_**
- **Weights** of YOLOv3 trained on Face-mask Dataset: [yolov3_face_mask.weights](https://bit.ly/yolov3_mask_weights)

**YOLOv4 Training details**

- Data File = [obj.data](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov4-mask-detector/obj.data)
- Cfg file  = [yolov4-obj.cfg](https://raw.githubusercontent.com/adityap27/face-mask-detector/master/yolov4-mask-detector/yolov4-obj.cfg)
- Pretrained Weights for initialization= [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)
- Main Configs from yolov4-obj.cfg:
	- learning_rate=0.001
	- batch=64
	- subdivisions=64
	- steps=4800,5400
	- max_batches = 6000
	- i.e approx epochs = (6000*64)/700 = 548
- **YOLOv4 Training results: _1.19 avg loss_**
- **Weights** of YOLOv4 trained on Face-mask Dataset: [yolov4_face_mask.weights](https://bit.ly/yolov4_mask_weights)

### 2. Model Performance
- Below is the comparison of YOLOv2, YOLOv3 and YOLOv4 on 3 sets.
- **Metric is mAP@0.5** i.e Mean Average Precision.
- **Frames per Second (FPS)** was measured on **Google Colab GPU - Tesla P100-PCIE** using **Darknet** command: [link](https://github.com/AlexeyAB/darknet#how-to-evaluate-fps-of-yolov4-on-gpu)

| Model | Training Set | Validation Set | Test Set | FPS |
|:--:|:--:|:--:|:--:|:--:|
| [YOLOv2](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv2%20Performance.jpg?raw=true) | 83.83% | 74.50% | 78.95% | 45 FPS |
| [YOLOv3](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv3%20Performance.jpg?raw=true) | 99.75% | 87.16% | 90.18% | 23 FPS |
| [YOLOv4](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv4%20Performance.jpg?raw=true) | 99.65% | 88.38% | 93.95% | 22 FPS |
- **Note:** For more detailed evaluation of model, click on model name above.
- **Conclusion:**
	- Yolov2 has **High bias** and **High Variance**, thus Poor Performance.
	- Yolov3 has **Low bias** and **Medium Variance**, thus Good Performance.
	- Yolov4 has **Low bias** and **Medium Variance**, thus Good Performance.
	- Model can still generalize well as discussed in section : [4. Suggestions to improve Performance](#Suggestions-to-improve-Performance)

### 3. Inference

- You can run model inference or detection on image/video/webcam.
- Two ways:
	1. Using Darknet itself
	2. Using Inference script (detection + alert)
- **Note:** If you are using yolov4 weights and cfg for inference, then make sure you use opencv>=4.4.0 else you will get ```Unsupported activation: mish ``` error.
### 3.1 Detection on Image
- Use command:
	```
	./darknet detector test obj.data yolov3.cfg yolov3_face_mask.weights input/1.jpg -thresh 0.45
	```
	OR
- Use inference script
	```
	python mask-detector-image.py -y yolov3-mask-detector -i input/1.jpg
	```
- **Output Image:**
	
	![1_output.jpg](https://github.com/adityap27/face-mask-detector/blob/master/output/1_output.jpg?raw=true)


### 3.2 Detection on Video
- Use command:
	```
	./darknet detector demo obj.data yolov3.cfg yolov3_face_mask.weights <video-file> -thresh 0.45
	```
	OR
- Use inference script
	```
	python mask-detector-video.py -y yolov3-mask-detector -i input/airport.mp4 -u 1
	```
	
- **Output Video:**
<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-airport.gif?raw=true">
</p>

### 3.3 Detection on WebCam
- Use command: (just remove input video file)
	```
	./darknet detector demo obj.data yolov3.cfg yolov3_face_mask.weights -thresh 0.45
	```
	OR
- Use inference script: (just remove input video file)
	```
	python mask-detector-video.py -y yolov3-mask-detector -u 1
	```
	
- **Output Video:**
<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-webcam.gif?raw=true">
</p>
	
### Note
- All the results(images & videos) shown are output of yolov3, you can use yolov4 for better results.

## Alert System
- Update:  E-mail notification support is added now as SMS are paid. 
- Alert system is present within the inference script code. 
- You can modify the SMS alert code in script to customize ratio for sms if you want.
- It monitors the mask, no-mask counts and has 3 status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.
	3. **Danger :** ( + SMS Alert ) When _some ratio_ of people are without mask.
<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-sms.jpg?raw=true">
</p>

## Suggestions to improve Performance
- As described earlier that yolov4 is giving 93.95% mAP on Test Set, this can be improved by following tips if you want:

	1. Use more Training Data.
	2. Use more Data Augmentation for Training Data.
	3. Train with larger network-resolution by setting your `.cfg-file` (height=640 and width=640) (any value multiple of 32).
	4. For Detection use even larger network-resolution like 864x864.
	5. Try YOLOv5 or any other Object Detection Algorithms like SSD, Faster-RCNN, RetinaNet, etc. as they are very good as of now (year 2020).



## References
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)
- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [Darknet github Repo](https://github.com/AlexeyAB/darknet)
- [YOLO Inference with GPU](https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/)


