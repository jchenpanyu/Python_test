# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:58:22 2017

http://www.runoob.com/python/python-exercise-example65.html
题目：一个最优美的图案。　　

@author: vincchen
"""

from Tkinter import *
import numpy as np
w = 800
h = 800
canvas = Canvas(width=w, height=h, bg='white')

# 外圆
canvas.create_oval(10, 10, w-10, h-10, fill="", outline="black")

#####
R = w/2 - 20
n_point = 19
delta_angle = np.pi*2/n_point
center_x = w/2
center_y = h/2
x = []
y = []
for i in np.arange(n_point):
    x.append(center_x + np.cos(i*delta_angle)*R)
    y.append(center_y + np.sin(i*delta_angle)*R)  
for i in np.arange(n_point):
    for j in np.arange(i+1, n_point):
        canvas.create_line(x[i], y[i], x[j], y[j])

canvas.pack()
mainloop()