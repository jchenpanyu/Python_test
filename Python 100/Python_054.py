# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:29:49 2017

http://www.runoob.com/python/python-exercise-example54.html
 Python 练习实例54
Python 100例 Python 100例
题目：取一个整数a从右端开始的4〜7位。
程序分析：可以这样考虑： 
(1)先使a右移4位。 
(2)设置一个低4位全为1,其余全为0的数。可用~(~0<<4) 
(3)将上面二者进行&运算。

@author: vincchen
"""

data = 4527
oper = 0b1111000
data_2 = (data & oper)

print bin(data)
print bin(data_2)