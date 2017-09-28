# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:46:15 2017
根据files的名字制备成标签
.pgm 
is the user id of the person in the image, and this field has 20 values: an2i, at33, boland, bpm, ch4f, cheyer, choon, danieln, glickman, karyadi, kawamura, kk49, megak, mitchell, night, phoebe, saavik, steffi, sz24, and tammo. 
is the head position of the person, and this field has 4 values: straight, left, right, up. 
is the facial expression of the person, and this field has 4 values: neutral, happy, sad, angry. 
is the eye state of the person, and this field has 2 values: open, sunglasses. 
is the scale of the image, and this field has 3 values: 1, 2, and 4. 1 indicates a full-resolution image (128 columns by 120 rows); 2 indicates a half-resolution image (64 by 60); 4 indicates a quarter-resolution image (32 by 30). 
If you've been looking closely in the image directories, you may notice that some images have a .bad suffix rather than the .pgm suffix. As it turns out, 16 of the 640 images taken have glitches due to problems with the camera setup; these are the .bad images. Some people had more glitches than others, but everyone who got ``faced'' should have at least 28 good face images (out of the 32 variations possible, discounting scale). 
@author: vincchen
"""

import os
import pandas as pd

path=r'.\data\faces'
face_attribute = ['name', 'head_position', 'facial_expression', 'eye_state', 'resolution']
df = pd.DataFrame(columns=face_attribute)

for root, dirs, files in os.walk(path):
    for f in files:
        if os.path.splitext(f)[1] == '.pgm':
            file_name = os.path.splitext(f)[0]
            file_split = file_name.split('_')
            if len(file_split) == 5:
                df = df.append(pd.DataFrame([file_split], columns=face_attribute))
            else:
                file_split.append(1) 
                df = df.append(pd.DataFrame([file_split], columns=face_attribute))
df.to_csv(r'.\data\face_label.txt', index=False,sep=',')