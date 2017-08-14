# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:03:53 2017

http://www.runoob.com/python/python-exercise-example3.html
Python 练习实例3
Python 100例 Python 100例
题目：一个整数，它加上100后是一个完全平方数，再加上168又是一个完全平方数，请问该数是多少？
程序分析：
假设该数为 x。
1、则：x + 100 = n^2, x + 100 + 168 = m^2
2、计算等式：m^2 - n^2 = (m + n)(m - n) = 168
3、设置： m + n = i，m - n = j，i * j =168，i 和 j 至少一个是偶数
4、可得： m = (i + j) / 2， n = (i - j) / 2，i 和 j 要么都是偶数，要么都是奇数。
5、从 3 和 4 推导可知道，i 与 j 均是大于等于 2 的偶数。
6、由于 i * j = 168， j>=2，则 1 < i < 168 / 2 + 1。
7、接下来将 i 的所有数字循环计算即可。


@author: vincchen
"""

import numpy as np

i = 0
# 找出小于 1000 的符合上述要求的数
while i <= 1000:
    a = i + 100
    b = i + 100 + 168
    if np.floor(np.sqrt(a)) == np.sqrt(a) and np.floor(np.sqrt(b)) == np.sqrt(b) :
        print i
    i = i + 1