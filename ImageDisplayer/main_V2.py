#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Image Displayer GUI
@time: 2018/2/23
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""
import numpy as np
import Tkinter as tk
from PIL import Image, ImageTk


WINDOW_WIDTH = 800 # 窗口宽度
WINDOW_HEIGHT = 400 # 窗口高度
WINDOW_X_OFFSET = 50
WINDOW_Y_OFFSET = 50
# 窗口大小,位置 # width x height + x_offset + y_offset:
WINDOW_SIZE = str(WINDOW_WIDTH) + "x" + str(WINDOW_HEIGHT) \
              + "+" + str(WINDOW_X_OFFSET) + "+" + str(WINDOW_Y_OFFSET)
WINDOW_TITLE = "Image Displayer" # 窗口题目
DEFAULT_IMAGE = r"C:\Users\vincchen\Documents\7_Coding\Python\Projects\ImageDisplayer\test.jpg" # 预设图片
DISPLAY_W = WINDOW_WIDTH/2-50
DISPLAY_H = WINDOW_HEIGHT-50

def printHello():
    print "hello"
    
def scale_image(img):
    img_w = float(img.size[0])
    img_h = float(img.size[1])
    if img_w/img_h >= DISPLAY_W/DISPLAY_H:
        scale_w = DISPLAY_W
        scale_h = int(img_h * (DISPLAY_W/img_w))
    else:
        scale_w = int(img_w * (DISPLAY_H/img_h))
        scale_h = DISPLAY_H
    return img.resize((scale_w, scale_h))
    
def image_to_array(img):
    return np.array(img)    

def array_to_image(img_array):
    r = Image.fromarray(img_array[:,:,0]).convert('L')
    g = Image.fromarray(img_array[:,:,1]).convert('L')
    b = Image.fromarray(img_array[:,:,2]).convert('L')
    return Image.merge("RGB", (r, g, b))

def imageProcess(image_Object):
    image_array = image_to_array(image_Object)
    image_array = image_array - 50
    return array_to_image(image_array)

def main():
    # 创建顶层窗口对象
    rootFrame = tk.Tk() 
    # 顶层窗口大小, Geometry=widthxheight+x+y, 前两个参数是窗口的宽度和高度。 最后两个参数是x，y 屏幕坐标。
    rootFrame.geometry(WINDOW_SIZE) 
    rootFrame.title(WINDOW_TITLE) # 设置窗口标题
    rootFrame.resizable(width=False, height=False) # 固定长宽不可拉伸
    # 设置展示的图片
    Default_img = Image.open(DEFAULT_IMAGE) 
    process_img = imageProcess(Default_img)
    displayImg_Left = ImageTk.PhotoImage(scale_image(Default_img))  
    displayImg_Right = ImageTk.PhotoImage(scale_image(process_img))
    # 创建次级frame装载image displayer跟按钮
    frame_top = tk.Frame(rootFrame) # 次级frame for image holder
    frame_bottom = tk.Frame(rootFrame) #次级frame for botton holder
    frame_top.pack(side='top')
    frame_bottom.pack(side='bottom')
    # 设置的default image displayer的布局
    frm_display_L = tk.Frame(frame_top)    
    tk.Label(frm_display_L, image=displayImg_Left, bg='white',
             width=DISPLAY_W, height=DISPLAY_H).pack(padx=25)
    frm_display_L.pack(side='left')
    # 设置的processing image displayer的布局
    frm_display_R = tk.Frame(frame_top)
    tk.Label(frm_display_R, image=displayImg_Right, bg='white',
             width=DISPLAY_W, height=DISPLAY_H).pack(padx=25)
    frm_display_R.pack(side='right')
    # 设置button的布局
    buttonFrame = tk.Frame(frame_bottom)
    tk.Button(buttonFrame, text="Load Image", command=printHello).pack(side='left')
    buttonFrame.pack(side='bottom')
    # 进入消息循环
    rootFrame.mainloop() 

if __name__ == '__main__':
    main()





