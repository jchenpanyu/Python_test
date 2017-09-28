# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:26:23 2017

把人脸截取出来作为数据源
用 up, straight & 1

@author: vincent
"""

import cv2
import os
import numpy as np

def face_rec(image, crop_size):
    classifier = cv2.CascadeClassifier(r'.\opencv\haarcascade_frontalface_default.xml')
    faceRects = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE)
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            center_x = x+w/2
            center_y = y+h/2
            crop_image = image[center_y-crop_size/2:center_y+crop_size/2, 
                               center_x-crop_size/2:center_x+crop_size/2]
            return crop_image
    else:
        return np.array(0)

input_path = r'.\data\faces'
output_path = r'.\data\save_faces'

for root, dirs, files in os.walk(input_path):
    for f in files:
        if os.path.splitext(f)[1] == '.pgm':
            file_name = os.path.splitext(f)[0]
            file_split = file_name.split('_')
            if len(file_split) == 4 and ( file_split[1] == 'straight' or file_split[1] == 'up'):
                image = cv2.imread(os.path.join(root, f))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                new_iamge = face_rec(image, 50)
                if new_iamge.size > 1:
                    cv2.imwrite(os.path.join(output_path, file_name+'_face.jpg'), new_iamge)










