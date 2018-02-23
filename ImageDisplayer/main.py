#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Image Displayer GUI
@time: 2018/2/23
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""
import Tkinter as tk
from PIL import Image, ImageTk

def printHello():
    print "hello"

def main():
    rootFrame = tk.Tk() # 创建窗口对象
    rootFrame.geometry('800x400') # 顶层窗口大小
    rootFrame.title("Image Displayer") # 窗口题目
    rootFrame.resizable(width=False, height=False) # 固定长宽不可拉伸
    
    img = Image.open(r"D:\Document\Python\Project\ImageDisplayer\test.jpg")
    displayImg = ImageTk.PhotoImage(img)
    
    frame_top = tk.Frame(rootFrame) # 次级frame for image holder
    frm_L = tk.Frame(frame_top)
    imgLabel=tk.Label(frm_L, image=displayImg, width=400, height=300)
    imgLabel.pack()
    frm_L.pack(side='left')    
    frm_R = tk.Frame(frame_top)
    imgLabel2=tk.Label(frm_R, image=displayImg, width=400, height=300)
    imgLabel2.pack()
    frm_R.pack(side='right')
    frame_top.pack()
    
    frame_bottom = tk.Frame(rootFrame) #次级frame for botton holder
    loadButton = tk.Button(frame_bottom, text="Load Image", command=printHello)
    loadButton.pack(side='bottom')
    frame_bottom.pack()
    
    rootFrame.mainloop() # 进入消息循环

if __name__ == '__main__':
    main()