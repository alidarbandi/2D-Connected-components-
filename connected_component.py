# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:33:22 2022

@author: sem
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#### Directory to load and save images. 
in_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet segment/save/connected components/original/'

out_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet segment/save/connected components/cc/'

area_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet segment/save/connected components/area/'

### if you have multiple images, write a for loop to go over each image and save the connected components information individually

for files in os.listdir(in_dir):
    path=in_dir+files
    if files.endswith('tiff'):  ### we are reading only tiff files
      print(path)
## read the image  
      image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)


## calculate connected components
      output=cv2.connectedComponentsWithStats(image,4, cv2.CV_8U)

### create a cc RGB image 
      number_labels, labels_info = cv2.connectedComponents(image)
      label_hue = np.uint8(179*labels_info/np.max(labels_info))
      blank_ch = 255*np.ones_like(label_hue)
      labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

      labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
      labeled_img[label_hue==0] = 0

### make a copy of the labelled image
      boxed=labeled_img.copy()

      num_labels = output[0]
# The second cell is the label matrix
      pixel_map = output[1]
# The third cell is the stat matrix
      stats = output[2]
# The fourth cell is the centroid matrix
      centroids = output[3]

### calculate each label area 
      area_list=np.zeros((number_labels,2),dtype=float)

      for i in range(1,number_labels): 
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]           
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
     
     ### area in um2 for pixel dimension of 5nm by 5nm 
        marea=area*0.25*0.0001
        area_list[i,:]=(int(i),marea)
     #print(area)
     
     ### draw a ractangle around each label
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
     ## annotate each label
        text=f"{i}"
        cv2.putText(boxed, text, (int(cX), int(cY)), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (255,225,255), thickness=1)
    
     ### save the final annotated image in the directoy
      saveName=out_dir+files
      cv2.imwrite(saveName,boxed)
 
### uncomment next line to save the txt file of list of labelled components and their corresponding area
      saveArea=area_dir+files[:-4]+"txt"
      np.savetxt(saveArea,area_list,delimiter='\t')
     
#### plot figures 


plt.subplot(2, 1, 1)
plt.imshow(image)
plt.title("Original Segmented Image")
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(boxed)
plt.title("Image after Component Labeling")
plt.show


