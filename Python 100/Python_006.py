# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:14:32 2017

http://www.runoob.com/python/python-exercise-example6.html
Python 练习实例6
Python 100例 Python 100例
题目：斐波那契数列。
程序分析：斐波那契数列（Fibonacci sequence），又称黄金分割数列，指的是这样一个数列：0、1、1、2、3、5、8、13、21、34、……。
在数学上，费波那契数列是以递归的方法来定义：

@author: vincchen
"""

import numpy as np

Fibonacci = [0, 1]

# 打印前10项
for i in np.arange(1, 10):
    Fibonacci.append(Fibonacci[i-1] + Fibonacci[i])

print 'Fibonacci sequence:', Fibonacci