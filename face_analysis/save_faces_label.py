# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:46:15 2017

@author: vincchen
"""

import os
import pandas as pd

path=r'.\data\save_faces'
face_attribute = ['name', 'head_position', 'facial_expression', 'eye_state']
df = pd.DataFrame(columns=face_attribute)

for root, dirs, files in os.walk(path):
    for f in files:
        file_name = os.path.splitext(f)[0]
        file_split = file_name.split('_')
        df = df.append(pd.DataFrame([file_split[0:4]], columns=face_attribute))
df.to_csv(r'.\data\save_faces_label.txt', index=False, sep=',')