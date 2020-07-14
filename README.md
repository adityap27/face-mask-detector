


# face-mask-detector
𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐝𝐞𝐞𝐩𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐀𝐥𝐞𝐫𝐭 𝐬𝐲𝐬𝐭𝐞𝐦 💻🔔


## System Overview

It detects human faces with 𝐦𝐚𝐬𝐤 𝐨𝐫 𝐧𝐨-𝐦𝐚𝐬𝐤 even in crowd in real time with live count status and notifies user (officer) if danger.

<p align="center">
  <img src="https://github.com/adityap27/face-mask-detector/blob/master/media/readme-airport.gif?raw=true">
</p>

**System Modules:**
  
1. **Deep Learning Model :** I trained a YOLOv2 and v3 on my own dataset and for YOLOv3 achieved **91% mAP on Test Set** even though my test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.  
  
2. **Alert System:** It monitors the mask, no-mask counts and has 3 status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.
	3. **Danger :** ( + SMS Alert ) When _some ratio_ of people are without mask.


## Table of Contents
1. [Face-Mask Dataset](#Face-Mask-Dataset)
	1. [Image Sources](#1.-Image-Sources)
	2. [Image Annotation](#2.-Image-Annotation) 
	3. [Dataset Description](#3.-Dataset-Description)
2. Deep Learning Models
	1. Training
	2. Model Performance 
	3. Inference 
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
