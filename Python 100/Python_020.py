# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:48:57 2017

http://www.runoob.com/python/python-exercise-example20.html
Python 练习实例20
Python 100例 Python 100例
题目：一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？
程序分析：无

@author: vincchen
"""

intial_h = 100.0
n_HitGound = 10
H = [intial_h]

for i in range(1, n_HitGound+1):
   H.append(H[i-1] / 2)

print zip(range(0, 11), H)