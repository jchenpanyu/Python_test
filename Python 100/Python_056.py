# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:52:06 2017

http://www.runoob.com/python/python-exercise-example56.html
Python 练习实例56
Python 100例 Python 100例
题目：画图，学用circle画圆形。　　　


@author: vincchen
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
if __name__ == '__main__':
    from Tkinter import *
 
    canvas = Canvas(width=800, height=600, bg='yellow')  
    canvas.pack(expand=YES, fill=BOTH)                
    k = 1
    j = 1
    for i in range(0,26):
        canvas.create_oval(310 - k,250 - k,310 + k,250 + k, width=1)
        k += j
        j += 0.3
 
    mainloop()