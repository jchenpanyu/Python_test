# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:22:31 2017

http://www.runoob.com/python/python-exercise-example9.html
Python 练习实例9
Python 100例 Python 100例
题目：暂停一秒输出。
程序分析：使用 time 模块的 sleep() 函数。

@author: vincchen
"""

import time
 
myD = {1: 'a', 2: 'b'}
for key, value in dict.items(myD):
    print key, value
    time.sleep(1) # 暂停 1 秒