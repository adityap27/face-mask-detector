
# face-mask-detector
𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐝𝐞𝐞𝐩𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐀𝐥𝐞𝐫𝐭 𝐬𝐲𝐬𝐭𝐞𝐦 💻🔔


## System Overview

It detects human faces with 𝐦𝐚𝐬𝐤 𝐨𝐫 𝐧𝐨-𝐦𝐚𝐬𝐤 even in crowd in real time with live count status and notifies user(officer) if danger.

### System Modules:
  
**1.) Deep Learning Model :** I trained a YOLOv2 and v3 on my own dataset and for YOLOv3 achieved **91%mAP on Test Set** even though my test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.  
  
**2.) Alert System:** It monitors the mask, no-mask counts and has 3 status :\
&nbsp;&nbsp;&nbsp; **i) Safe**\
&nbsp;&nbsp;&nbsp; **ii) Warning**\
&nbsp;&nbsp;&nbsp; **iii) Danger** ( SMS Alert )
