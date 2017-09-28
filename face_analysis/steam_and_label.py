# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:08:25 2017

steam the face and label

label:
    neutral: 0
    happy: 1
    sad: 2
    angry: 3

@author: vincent
"""

import os
import cv2
import numpy as np

path = r'.\data\save_faces'
facial_label = {'neutral':0, 'happy':1, 'sad':2, 'angry':3}
data_steam = []
for root, dirs, files in os.walk(path):
    for f in files:
        image = cv2.imread(os.path.join(root, f))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # append image steam and facial expression label
        label = os.path.splitext(f)[0].split('_')[2]
        data_steam.append(np.append(image.flatten(), facial_label[label]))

out_path = r'.\data'      
np.savetxt(os.path.join(out_path, 'face_label.csv'), data_steam, delimiter=',')