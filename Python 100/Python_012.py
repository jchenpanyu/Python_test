#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
http://www.runoob.com/python/python-exercise-example12.html
Python 练习实例12
Python 100例 Python 100例
题目：判断101-200之间有多少个素数，并输出所有素数。
程序分析：判断素数的方法：用一个数分别去除2到sqrt(这个数)，如果能被整除，则表明此数不是素数，反之是素数
"""

import numpy as np

prime_number = []

for n in range(101, 201):
    for i in range(2, int(np.floor(np.sqrt(n)))+2):
        if n % i == 0:
            break
        if i == np.floor(np.sqrt(n))+ 1:
            prime_number.append(n)

print prime_number
print len(prime_number)