# -*- coding: utf-8 -*-

"""
Created on Sat Sep 30 11:13:06 2017
@author: vincchen
E-mail: vincentchan.sysu@gmail.com

本函数用于把输入的图片(image)中，人脸部分提取出去，并把人脸的尺寸归一化到crop_size
目前仅适用于image中只有一张或没有人脸的情况
对于image中有多张人脸的情况，有待进一步开发
"""

import cv2
import numpy as np

def face_crop(image, crop_size):
    # image为待处理的图片
    # crop_size为从image截取人脸(截取为正方形)后的缩放的尺寸，数据类型为int，例如50
    
    # 采用opencv下的人脸识别classifier
    classifier = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml')
    face = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE)
    # 判读是否有识别去人脸
    # Yes -> 对识别出来的人脸进行处理
    # No -> 返回一个crop_size大小的0 array
    if len(face) > 0:
        for face_coordinate in face:
            # x, y为人脸识别框的左上角坐标
            # w, h为人脸识别框的宽度，高度
            x, y, w, h = face_coordinate
            # 为保证人脸截取框为正方形，选择w, h较少的一边作为截取框边长，以防边长会溢出
            L = np.min(w, h)    
            # 提取人脸部分的image
            cropface_image = image[y:y+L, x:x+L]
            # 把cropface_image resacle到crop_size
            # interpolation - 插值方法。共有5种：
            # 1) INTER_NEAREST - 最近邻插值法
            # 2) INTER_LINEAR - 双线性插值法（默认）
            # 3) INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。
            #   对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
            # 4) INTER_CUBIC - 基于4x4像素邻域的3次插值法
            # 5) INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
            ratio = float(crop_size)/L
            resize_image = cv2.resize(cropface_image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            return resize_image
    else:
        return np.zeros((crop_size, crop_size))
