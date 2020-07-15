




# face-mask-detector
𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐝𝐞𝐞𝐩𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐀𝐥𝐞𝐫𝐭 𝐬𝐲𝐬𝐭𝐞𝐦 💻🔔


## System Overview

It detects human faces with 𝐦𝐚𝐬𝐤 𝐨𝐫 𝐧𝐨-𝐦𝐚𝐬𝐤 even in crowd in real time with live count status and notifies user (officer) if danger.

<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-airport.gif?raw=true">
</p>

**System Modules:**
  
1. **Deep Learning Model :** I trained a YOLOv2 and v3 on my own dataset and for YOLOv3 achieved **90% mAP on Test Set** even though my test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.  
  
2. **Alert System:** It monitors the mask, no-mask counts and has 3 status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.
	3. **Danger :** ( + SMS Alert ) When _some ratio_ of people are without mask.


## Table of Contents
1. [Face-Mask Dataset](#Face-Mask-Dataset)
	1. [Image Sources](#1.-Image-Sources)
	2. [Image Annotation](#2.-Image-Annotation) 
	3. [Dataset Description](#3.-Dataset-Description)
2. [Deep Learning Models](#Deep-Learning-Models)
	1. [Training](#1.-Training)
	2. [Model Performance](#2.-Model-Performance)
	3. [Inference](#3.-Inference)
		1. [Detection on Image](#3.1-Detection-on-Image)
		2. [Detection on Video]()
		3. [Detection on WebCam]()
3. Alert System
4. Suggestions to improve Performance
5. References

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
- I have trained Yolov2 and Yolov3 both.
- Use following (linux) cmd to train:


```console
./darknet detector train obj.data yolo3.cfg darknet53.conv.74
```
- for windows use **darknet.exe** instead of ./darknet

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

### 2. Model Performance
- Below is the comparison of YOLOv2 and YOLOv3 on 3 sets.
- **Metric is mAP@0.5** i.e Mean Average Precision.

| Model | Training Set | Validation Set | Test Set |
|:--:|:--:|:--:|:--:|
| [YOLOv2](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv2%20Performance.jpg?raw=true) | 76.35% | 72.96% | 67.63% |
| [YOLOv3](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv3%20Performance.jpg?raw=true) | 99.75% | 87.16% | 90.18% |
- **Note:** For more detailed evaluation of model, click on model name above.
- **Conclusion:**
	- Yolov2 has **High bias** and **High Variance**, thus Poor Performance.
	- Yolov3 has **Low bias** and **Medium Variance**, thus Good Performance.
	- Model can still generalize well as discussed in section : [4. Suggestions to improve Performance]()

### 3. Inference

- You can run model inference or detection on image/video/webcam.
- Two ways:
	1. Using Darknet itself
	2. Using Inference script (detection + alert)

### 3.1 Detection on Image
- Use command:
	```
	./darknet detector test obj.data yolov3.cfg yolov3_face_mask.weights input/1.jpg
	```
	OR
- Use inference script
	```
	python mask-detector-image.py -y yolov3-mask-detector -i input\1.jpg
	```
- **Output Image:**
	
	![1_output.jpg](https://github.com/adityap27/face-mask-detector/blob/master/output/1_output.jpg?raw=true)
		 
