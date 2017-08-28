# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:46:30 2017

http://www.runoob.com/python/python-exercise-example64.html
Python 练习实例64
Python 100例 Python 100例
题目：利用ellipse 和 rectangle 画图。。

@author: vincchen
"""

from Tkinter import *

width = 400
height = 600
canvas = Canvas(width=width, height=height, bg='white')

# rectangle
L = 20
U = 20
R = width/2 - 20
D = height/2 - 20

# ellipse
L_1 = width/2 + 20
U_1 = height/2 + 20
R_1 = width - 20
D_1 = height - 20 

n= 20
for i in range(n):
    canvas.create_rectangle(L+2*i, U+2*i, R-5*i, D-5*i)
    canvas.create_oval(L_1+4*i, U_1+4*i, R_1-4*i, D_1-4*i)

canvas.pack()
mainloop()